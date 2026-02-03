#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gym
from torch import distributions as torchd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src_dreamerv3 import tools
from src_dreamerv3 import models


def load_config(config_names, overrides):
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    yaml = YAML(typ="safe", pure=True)
    configs = yaml.load(
        (Path(__file__).resolve().parents[2] / "src_dreamerv3" / "configs.yaml").read_text()
    )
    defaults = {}
    for name in ["defaults", *config_names]:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(overrides)


def pick_episode(episodes_dir, explicit_path=None):
    if explicit_path:
        return Path(explicit_path)
    candidates = list(Path(episodes_dir).glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No episodes found in {episodes_dir}")

    def ep_len(path):
        try:
            return int(path.stem.split("-")[-1])
        except ValueError:
            return 0

    return max(candidates, key=ep_len)


def infer_task_from_logdir(logdir):
    name = Path(logdir).name
    if "atari_" in name:
        suffix = name.split("atari_", 1)[1]
        task = suffix.split("_", 1)[0]
        return f"atari_{task}"
    return None


def compute_abs_kl_series(wm, data):
    wm.eval()
    with torch.no_grad():
        proc = wm.preprocess(data)
        embed = wm.encoder(proc)
        reward = proc.get("reward", None)
        post, prior = wm.dynamics.observe(embed, proc["action"], proc["is_first"], reward=reward)

        post_abs = torchd.normal.Normal(post["abs_mean"], post["abs_std"])
        prior_abs = torchd.normal.Normal(prior["abs_mean"], prior["abs_std"])

        abs_kl_post_prior = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_abs, 1),
            torchd.independent.Independent(prior_abs, 1),
        )
        abs_kl_prior_post = torchd.kl.kl_divergence(
            torchd.independent.Independent(prior_abs, 1),
            torchd.independent.Independent(post_abs, 1),
        )

        post_boundary_logit = post.get("boundary_logit", None)
        prior_boundary_logit = prior.get("boundary_logit", None)
        if post_boundary_logit is None or prior_boundary_logit is None:
            kl_mask = None
        else:
            post_boundary_prob = F.softmax(post_boundary_logit, dim=-1)
            prior_boundary_prob = F.softmax(prior_boundary_logit, dim=-1)
            eps = 1e-8
            kl_mask = (
                post_boundary_prob
                * (torch.log(post_boundary_prob + eps) - torch.log(prior_boundary_prob + eps))
            ).sum(dim=-1)

    abs_kl_np = abs_kl_post_prior.detach().cpu().numpy()
    abs_kl_rev_np = abs_kl_prior_post.detach().cpu().numpy()
    kl_mask_np = kl_mask.detach().cpu().numpy() if kl_mask is not None else None
    return abs_kl_np, abs_kl_rev_np, kl_mask_np  # (B, T)


def moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", nargs="+", required=True, help="One or more logdirs.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names (one per logdir). If omitted, inferred from logdir name.",
    )
    parser.add_argument(
        "--ckpt_paths",
        nargs="+",
        default=None,
        help="Checkpoint paths (one per logdir). Defaults to logdir/latest.pt.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Episode .npz paths (one per logdir). Defaults to longest in episodes_dir.",
    )
    parser.add_argument("--episodes_dir", default="train_eps")
    parser.add_argument("--configs", nargs="+", default=["atari100k"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Moving average window for visualization. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--align",
        choices=["min", "full"],
        default="min",
        help="Align series length to min length or keep full with NaNs.",
    )
    parser.add_argument(
        "--out",
        default="vta_abs_kl_plot.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--out_ctx",
        default="vta_ctx_kl_plot.png",
        help="Output plot path for context KL only.",
    )
    parser.add_argument(
        "--csv_out",
        default="vta_abs_kl_series.csv",
        help="Output CSV for per-step KL series.",
    )
    parser.add_argument(
        "--out_abs_bidirectional",
        default="vta_abs_kl_bidirectional.png",
        help="Output plot path for abs KL in both directions.",
    )
    args, overrides = parser.parse_known_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logdirs = [Path(p) for p in args.logdirs]
    num = len(logdirs)

    tasks = args.tasks or [infer_task_from_logdir(p) for p in logdirs]
    if any(t is None for t in tasks):
        missing = [str(logdirs[i]) for i, t in enumerate(tasks) if t is None]
        raise ValueError(
            "Task inference failed for: "
            + ", ".join(missing)
            + ". Provide --tasks explicitly."
        )
    if len(tasks) != num:
        raise ValueError("--tasks must have the same length as --logdirs.")

    ckpt_paths = args.ckpt_paths or [None] * num
    if len(ckpt_paths) != num:
        raise ValueError("--ckpt_paths must have the same length as --logdirs.")

    episodes = args.episodes or [None] * num
    if len(episodes) != num:
        raise ValueError("--episodes must have the same length as --logdirs.")

    series = []
    series_rev = []
    ctx_series = []
    meta = []

    for logdir, task, ckpt_path, episode in zip(logdirs, tasks, ckpt_paths, episodes):
        config = load_config(
            args.configs,
            ["--task", task, "--dynamics_type", "vta", "--device", device, *overrides],
        )

        ckpt = Path(ckpt_path) if ckpt_path else (logdir / "latest.pt")
        if not ckpt.is_absolute():
            ckpt = logdir / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        ep_path = pick_episode(logdir / args.episodes_dir, episode)
        with np.load(ep_path) as ep:
            images = ep["image"]
            actions = ep["action"]
            reward = ep["reward"] if "reward" in ep else None
            is_first = ep["is_first"]
            is_terminal = ep["is_terminal"]
            discount = ep["discount"] if "discount" in ep else None

        action_dim = actions.shape[-1]
        config.num_actions = action_dim

        obs_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, shape=images.shape[1:], dtype=np.uint8)}
        )
        act_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
        act_space.discrete = True

        wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(device)
        state = torch.load(ckpt, map_location=device)["agent_state_dict"]
        wm_state = {k[len("_wm."):]: v for k, v in state.items() if k.startswith("_wm.")}
        wm_state = {k.replace("_orig_mod.", ""): v for k, v in wm_state.items()}
        wm.load_state_dict(wm_state, strict=True)

        data = {
            "image": images[None],
            "action": actions[None],
            "is_first": is_first[None],
            "is_terminal": is_terminal[None],
        }
        if reward is not None:
            data["reward"] = reward[None]
        if discount is not None:
            data["discount"] = discount[None]

        abs_kl, abs_kl_rev, kl_mask = compute_abs_kl_series(wm, data)
        abs_kl = abs_kl[0]  # (T,)
        abs_kl_rev = abs_kl_rev[0]  # (T,)
        if kl_mask is None:
            raise RuntimeError(
                "Context KL (kl_mask) not found. "
                "This run may not be using VTA with boundary logits."
            )
        kl_mask = kl_mask[0]  # (T,)
        series.append(abs_kl)
        series_rev.append(abs_kl_rev)
        ctx_series.append(kl_mask)
        meta.append(
            {
                "logdir": str(logdir),
                "task": task,
                "ckpt": str(ckpt),
                "episode": str(ep_path),
            }
        )

    lengths = [len(s) for s in series]
    if args.align == "min":
        T = min(lengths)
        series = [s[:T] for s in series]
        series_rev = [s[:T] for s in series_rev]
        ctx_series = [s[:T] for s in ctx_series]
        t = np.arange(T)
    else:
        T = max(lengths)
        t = np.arange(T)
        aligned = []
        aligned_rev = []
        aligned_ctx = []
        for s in series:
            if len(s) == T:
                aligned.append(s)
            else:
                pad = np.full((T - len(s),), np.nan, dtype=np.float32)
                aligned.append(np.concatenate([s, pad], axis=0))
        series = aligned
        for s in series_rev:
            if len(s) == T:
                aligned_rev.append(s)
            else:
                pad = np.full((T - len(s),), np.nan, dtype=np.float32)
                aligned_rev.append(np.concatenate([s, pad], axis=0))
        series_rev = aligned_rev
        for s in ctx_series:
            if len(s) == T:
                aligned_ctx.append(s)
            else:
                pad = np.full((T - len(s),), np.nan, dtype=np.float32)
                aligned_ctx.append(np.concatenate([s, pad], axis=0))
        ctx_series = aligned_ctx

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "logdir",
                "task",
                "ckpt",
                "episode",
                "t",
                "abs_kl_post_prior",
                "abs_kl_prior_post",
                "ctx_kl",
            ],
        )
        writer.writeheader()
        for s, sr, c, m in zip(series, series_rev, ctx_series, meta):
            for i, v in enumerate(s):
                writer.writerow(
                    {
                        "logdir": m["logdir"],
                        "task": m["task"],
                        "ckpt": m["ckpt"],
                        "episode": m["episode"],
                        "t": i,
                        "abs_kl_post_prior": float(v) if np.isfinite(v) else float("nan"),
                        "abs_kl_prior_post": float(sr[i]) if np.isfinite(sr[i]) else float("nan"),
                        "ctx_kl": float(c[i]) if np.isfinite(c[i]) else float("nan"),
                    }
                )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for s, sr, c, m in zip(series, series_rev, ctx_series, meta):
        sm = moving_average(np.nan_to_num(s, nan=0.0), args.smooth)
        sr_sm = moving_average(np.nan_to_num(sr, nan=0.0), args.smooth)
        c_sm = moving_average(np.nan_to_num(c, nan=0.0), args.smooth)
        label = f"{m['task']} (mean={np.nanmean(s):.4f})"
        axes[0].plot(t, sm, linewidth=2.0, label=label)
        axes[0].plot(t, s, linewidth=1.0, alpha=0.25)

        label_rev = f"{m['task']} (mean={np.nanmean(sr):.4f})"
        axes[1].plot(t, sr_sm, linewidth=2.0, label=label_rev)
        axes[1].plot(t, sr, linewidth=1.0, alpha=0.25)

        label_ctx = f"{m['task']} (mean={np.nanmean(c):.4f})"
        axes[2].plot(t, c_sm, linewidth=2.0, label=label_ctx)
        axes[2].plot(t, c, linewidth=1.0, alpha=0.25)

    axes[0].set_ylabel("KL(post_z || prior_z) [abs]")
    axes[0].set_title("VTA abs z KL (posterior vs prior)")
    axes[0].legend(loc="upper right")

    axes[1].set_ylabel("KL(prior_z || post_z) [abs]")
    axes[1].set_title("VTA abs z KL (prior vs posterior)")
    axes[1].legend(loc="upper right")

    axes[2].set_ylabel("KL(q(b|x) || p(b|s))")
    axes[2].set_xlabel("Time Step")
    axes[2].set_title("Context KL (boundary posterior vs prior)")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Bidirectional KL plot for abs z
    fig, ax = plt.subplots(figsize=(10, 5))
    for s, sr, m in zip(series, series_rev, meta):
        sm = moving_average(np.nan_to_num(s, nan=0.0), args.smooth)
        sr_sm = moving_average(np.nan_to_num(sr, nan=0.0), args.smooth)
        label_pp = f"{m['task']} post||prior"
        label_pp_rev = f"{m['task']} prior||post"
        ax.plot(t, sm, linewidth=2.0, label=label_pp)
        ax.plot(t, sr_sm, linewidth=2.0, linestyle="--", label=label_pp_rev)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("KL (abs z)")
    ax.set_title("VTA abs z KL (posterior vs prior, both directions)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_abs_bi = Path(args.out_abs_bidirectional)
    out_abs_bi.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_abs_bi, dpi=150)
    plt.close(fig)

    # Optional: context-only plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for c, m in zip(ctx_series, meta):
        c_sm = moving_average(np.nan_to_num(c, nan=0.0), args.smooth)
        label_ctx = f"{m['task']} (mean={np.nanmean(c):.4f})"
        ax.plot(t, c_sm, linewidth=2.0, label=label_ctx)
        ax.plot(t, c, linewidth=1.0, alpha=0.25)
    ax.set_ylabel("KL(q(b|x) || p(b|s))")
    ax.set_xlabel("Time Step")
    ax.set_title("Context KL (boundary posterior vs prior)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_ctx_path = Path(args.out_ctx)
    out_ctx_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_ctx_path, dpi=150)
    plt.close(fig)

    summary = {
        "out": str(out_path),
        "csv": str(csv_path),
        "series_len": int(T),
        "align": args.align,
        "smooth": args.smooth,
        "out_ctx": str(out_ctx_path),
        "items": meta,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
