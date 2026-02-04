#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from ruamel.yaml import YAML
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import gymnasium as gym
except Exception:  # gymnasium not available
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
    # Prefer the longest episode (length encoded at filename suffix)
    def ep_len(path):
        try:
            return int(path.stem.split("-")[-1])
        except ValueError:
            return 0
    return max(candidates, key=ep_len)


def compute_vta_stats(wm, data):
    wm.eval()
    with torch.no_grad():
        proc = wm.preprocess(data)
        embed = wm.encoder(proc)
        reward = proc.get("reward", None)
        post, prior = wm.dynamics.observe(embed, proc["action"], proc["is_first"], reward=reward)
        
        # For post_boundary, also need to pass reward if embed_reward mode
        if hasattr(wm.dynamics, '_posterior_input') and wm.dynamics._posterior_input == 'embed_reward' and reward is not None:
            # Concatenate reward to embed for post_boundary
            reward_inp = reward.unsqueeze(-1)  # (batch, time, 1)
            post_inp = torch.cat([embed, reward_inp], dim=-1)
            post_logits = wm.dynamics.post_boundary(post_inp)
        else:
            post_logits = wm.dynamics.post_boundary(embed)
        post_probs = torch.softmax(post_logits, dim=-1)[..., 0]

        post_abs = torchd.normal.Normal(post["abs_mean"], post["abs_std"])
        prior_abs = torchd.normal.Normal(prior["abs_mean"], prior["abs_std"])
        post_obs = torchd.normal.Normal(post["obs_mean"], post["obs_std"])
        prior_obs = torchd.normal.Normal(prior["obs_mean"], prior["obs_std"])

        abs_kl = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_abs, 1),
            torchd.independent.Independent(prior_abs, 1),
        )
        obs_kl = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_obs, 1),
            torchd.independent.Independent(prior_obs, 1),
        )

    stats = {
        "boundary": post["boundary"].squeeze(-1).detach().cpu().numpy(),
        "read_prob": post_probs.detach().cpu().numpy(),
        "seg_len": post["seg_len"].squeeze(-1).detach().cpu().numpy(),
        "seg_num": post["seg_num"].squeeze(-1).detach().cpu().numpy(),
        "abs_kl": abs_kl.detach().cpu().numpy(),
        "obs_kl": obs_kl.detach().cpu().numpy(),
    }
    return stats


def choose_window(boundary, reward, frame_delta, length, start=None, mode="boundary"):
    if start is not None:
        return int(start)
    if boundary.shape[1] <= length:
        return 0
    if mode == "reward":
        if reward is None or np.allclose(reward, 0):
            mode = "boundary"
        else:
            scores = np.convolve(np.abs(reward), np.ones(length), mode="valid")
            return int(np.argmax(scores))
    if mode == "delta":
        if frame_delta is None:
            mode = "boundary"
        else:
            scores = np.convolve(frame_delta, np.ones(length), mode="valid")
            return int(np.argmax(scores))
    if mode == "reward_or_delta":
        if reward is not None and not np.allclose(reward, 0):
            scores = np.convolve(np.abs(reward), np.ones(length), mode="valid")
            return int(np.argmax(scores))
        if frame_delta is not None:
            scores = np.convolve(frame_delta, np.ones(length), mode="valid")
            return int(np.argmax(scores))
    # Pick window with most boundaries for clearer visualization.
    counts = np.convolve(boundary[0], np.ones(length), mode="valid")
    return int(np.argmax(counts))


def render_plot(images, boundary, out_path, title):
    t_len = len(images)
    fig = plt.figure(figsize=(max(8, t_len * 0.8), 5))
    gs = fig.add_gridspec(2, t_len, height_ratios=[3, 1.2], hspace=0.35)

    # Frames with colored borders indicating boundary.
    for t in range(t_len):
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(images[t])
        ax.set_title(f"t={t}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        color = "red" if boundary[t] > 0.5 else "blue"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)

    ax = fig.add_subplot(gs[1, :])
    t = np.arange(t_len)
    colors = ["#e41a1c" if b > 0.5 else "#377eb8" for b in boundary]
    ax.bar(t, boundary, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("m (boundary)")
    ax.set_xlabel("Time Step")
    ax.set_title(title)
    # Highlight READ steps with vertical bands for clarity.
    for idx, b in enumerate(boundary):
        if b > 0.5:
            ax.axvspan(idx - 0.45, idx + 0.45, color="#e41a1c", alpha=0.1)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#e41a1c", label="READ (boundary)"),
        plt.Rectangle((0, 0), 1, 1, color="#377eb8", label="COPY (continue)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_internal_plot(stats, reward, frame_delta, start, end, out_path, title):
    boundary = stats["boundary"][0, start:end]
    read_prob = stats["read_prob"][0, start:end]
    seg_len = stats["seg_len"][0, start:end]
    seg_num = stats["seg_num"][0, start:end]
    abs_kl = stats["abs_kl"][0, start:end]
    obs_kl = stats["obs_kl"][0, start:end]
    t = np.arange(len(boundary))

    reward_win = reward[start:end] if reward is not None else None
    delta_win = frame_delta[start:end] if frame_delta is not None else None
    if reward_win is not None:
        if torch.is_tensor(reward_win):
            reward_win = reward_win.detach().cpu().numpy()
        reward_win = np.asarray(reward_win).reshape(-1)
        if reward_win.shape[0] != t.shape[0]:
            reward_win = None
    if delta_win is not None:
        delta_win = np.asarray(delta_win).reshape(-1)
        if delta_win.shape[0] != t.shape[0]:
            delta_win = None
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, read_prob, color="black", linewidth=2.0, label="READ prob")
    axes[0].scatter(
        t,
        boundary,
        color=["#e41a1c" if b > 0.5 else "#377eb8" for b in boundary],
        s=30,
        zorder=3,
    )
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("READ prob / sample")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, seg_len, color="#1f77b4", label="seg_len")
    axes[1].plot(t, seg_num, color="#ff7f0e", label="seg_num")
    axes[1].set_ylabel("Segment")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, abs_kl, color="#2ca02c", label="abs_kl")
    axes[2].plot(t, obs_kl, color="#d62728", label="obs_kl")
    axes[2].set_ylabel("KL")
    axes[2].legend(loc="upper right")

    if reward_win is not None:
        axes[3].bar(t, reward_win, color="#9467bd", alpha=0.6, label="reward")
    if delta_win is not None:
        axes[3].plot(t, delta_win, color="#8c564b", linewidth=1.5, label="frame_delta")
    axes[3].set_ylabel("Reward / Delta")
    axes[3].set_xlabel("Time Step")
    axes[3].legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_fired_plot(images, boundary, indices, out_path, title):
    if len(indices) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No READ boundaries in this episode.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    cols = min(10, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig = plt.figure(figsize=(max(8, cols * 1.2), max(2.5, rows * 1.6)))
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.2)

    for i, idx in enumerate(indices):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(images[idx])
        ax.set_title(f"t={idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#e41a1c")
            spine.set_linewidth(2.0)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    def prepare_csv(path: Path, fieldnames, overwrite: bool):
        """
        Returns (final_path, write_header, mode).
        If an existing file has a different header, default to writing a new *_v2.csv file,
        unless overwrite=True.
        """
        if not path.exists():
            return path, True, "w"
        try:
            with path.open() as f:
                header = f.readline().strip().split(",")
        except OSError:
            header = []
        if header == list(fieldnames):
            return path, False, "a"
        if overwrite:
            return path, True, "w"
        v2 = path.with_name(f"{path.stem}_v2{path.suffix}")
        return v2, True, "w"

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Directory with checkpoint and episodes.")
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Checkpoint path (defaults to logdir/latest.pt). Relative paths are resolved from --logdir.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save outputs (defaults to --logdir).",
    )
    parser.add_argument(
        "--csv_out",
        default=None,
        help="CSV path to append summary stats (defaults to output_dir/boundary_stats.csv).",
    )
    parser.add_argument(
        "--dist_window",
        type=int,
        default=20,
        help="Window size for z_t to z_{t+1} distance summary.",
    )
    parser.add_argument(
        "--dist_stride",
        type=int,
        default=5,
        help="Stride for sliding window summaries.",
    )
    parser.add_argument(
        "--dist_csv",
        default=None,
        help="CSV path to write z_t/z_{t+1} distance summaries (defaults to output_dir/zt_zt_1.csv).",
    )
    parser.add_argument(
        "--dist_plot_out",
        default=None,
        help="Output image path for z_t/z_{t+1} distance plot (defaults to output_dir/zt_zt_1_plot.png).",
    )
    parser.add_argument(
        "--overwrite_csv",
        action="store_true",
        help="Overwrite CSV outputs if an incompatible schema is detected.",
    )
    parser.add_argument("--configs", nargs="+", default=["atari100k"])
    parser.add_argument("--task", default="atari_breakout")
    parser.add_argument("--episode", default=None, help="Path to a .npz episode file")
    parser.add_argument("--episodes_dir", default="train_eps")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument(
        "--window",
        default="boundary",
        choices=["boundary", "reward", "delta", "reward_or_delta"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--internal_out", default=None)
    parser.add_argument("--fired_out", default=None)
    args, overrides = parser.parse_known_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = load_config(
        args.configs,
        ["--task", args.task, "--dynamics_type", "vta", "--device", device, *overrides],
    )

    logdir = Path(args.logdir)
    output_dir = Path(args.output_dir) if args.output_dir else logdir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else (logdir / "latest.pt")
    if not ckpt_path.is_absolute():
        ckpt_path = logdir / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ep_path = pick_episode(logdir / args.episodes_dir, args.episode)
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
    state = torch.load(ckpt_path, map_location=device)["agent_state_dict"]
    wm_state = {k[len("_wm."):]: v for k, v in state.items() if k.startswith("_wm.")}
    # Remove _orig_mod. prefix from torch.compile() saved checkpoints
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

    stats = compute_vta_stats(wm, data)

    # Compute z (abstract) and s (obs) distances for adjacent steps.
    # Note:
    # - *_mean below refers to posterior distribution parameters (can change even if COPY).
    # - *_stoch below refers to the latent state actually used by the dynamics (COPY makes it piecewise constant).
    with torch.no_grad():
        proc = wm.preprocess(data)
        embed = wm.encoder(proc)
        reward = proc.get("reward", None)
        post, _ = wm.dynamics.observe(embed, proc["action"], proc["is_first"], reward=reward)
        abs_mean = post["abs_mean"].detach().cpu().numpy()  # (B, T, D)
        obs_mean = post["obs_mean"].detach().cpu().numpy()
        abs_stoch = post["abs_stoch"].detach().cpu().numpy()
        obs_stoch = post["obs_stoch"].detach().cpu().numpy()
        boundary_logit = post.get("boundary_logit", None)
        read_prob = (
            torch.softmax(boundary_logit, dim=-1)[..., 0].detach().cpu().numpy()
            if boundary_logit is not None
            else stats.get("read_prob", None)
        )
        boundary = stats["boundary"]  # (B, T)

    def adjacent_l2(x):
        diff = x[:, 1:, :] - x[:, :-1, :]
        return np.linalg.norm(diff, axis=-1)  # (B, T-1)

    abs_l2 = adjacent_l2(abs_mean)
    obs_l2 = adjacent_l2(obs_mean)
    abs_stoch_l2 = adjacent_l2(abs_stoch)
    obs_stoch_l2 = adjacent_l2(obs_stoch)
    boundary_next = boundary[:, 1:]  # align with z_t -> z_{t+1}
    read_prob_next = read_prob[:, 1:] if read_prob is not None else None

    abs_kl_mean = float(stats["abs_kl"].mean())
    abs_kl_std = float(stats["abs_kl"].std())
    obs_kl_mean = float(stats["obs_kl"].mean())
    obs_kl_std = float(stats["obs_kl"].std())
    boundary_rate = float((stats["boundary"] > 0.5).mean())
    read_prob_mean = float(np.asarray(read_prob).mean()) if read_prob is not None else float("nan")
    read_prob_p99 = (
        float(np.quantile(np.asarray(read_prob).reshape(-1), 0.99)) if read_prob is not None else float("nan")
    )

    frame_delta = None
    if images.shape[0] > 1:
        diff = np.abs(images[1:].astype(np.float32) - images[:-1].astype(np.float32))
        frame_delta = diff.mean(axis=(1, 2, 3))
        frame_delta = np.concatenate([[0.0], frame_delta], axis=0)

    start = choose_window(stats["boundary"], reward, frame_delta, args.length, args.start, args.window)
    end = start + args.length

    images_win = images[start:end]
    boundary_win = stats["boundary"][0, start:end]

    out_path = Path(args.out) if args.out else output_dir / "boundary_grid.png"
    title = f"VTA Boundary Detection - {args.task} ({ep_path.name})"
    render_plot(images_win, boundary_win, out_path, title)
    print(out_path)

    internal_out = (
        Path(args.internal_out) if args.internal_out else output_dir / "boundary_internal.png"
    )
    render_internal_plot(stats, reward, frame_delta, start, end, internal_out, title)
    print(internal_out)

    fired_out = Path(args.fired_out) if args.fired_out else output_dir / "boundary_fired.png"
    boundary_all = stats["boundary"][0]
    fired_indices = np.where(boundary_all > 0.5)[0].tolist()
    render_fired_plot(images, boundary_all, fired_indices, fired_out, title)
    print(fired_out)

    summary = {
        "task": args.task,
        "logdir": str(logdir),
        "ckpt": str(ckpt_path),
        "episode": str(ep_path),
        "abs_kl_mean": abs_kl_mean,
        "abs_kl_std": abs_kl_std,
        "obs_kl_mean": obs_kl_mean,
        "obs_kl_std": obs_kl_std,
        "boundary_rate": boundary_rate,
        "read_prob_mean": read_prob_mean,
        "read_prob_p99": read_prob_p99,
        "abs_l2_mean": float(abs_l2.mean()),
        "obs_l2_mean": float(obs_l2.mean()),
        "abs_stoch_l2_mean": float(abs_stoch_l2.mean()),
        "obs_stoch_l2_mean": float(obs_stoch_l2.mean()),
        "abs_l2_read_mean": float(abs_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
        "obs_l2_read_mean": float(obs_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
        "abs_stoch_l2_read_mean": float(abs_stoch_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
        "obs_stoch_l2_read_mean": float(obs_stoch_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
    }
    print(json.dumps(summary, ensure_ascii=False))

    csv_path = Path(args.csv_out) if args.csv_out else output_dir / "boundary_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_fieldnames = list(summary.keys())
    csv_path, write_header, mode = prepare_csv(csv_path, summary_fieldnames, args.overwrite_csv)
    with csv_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    # Sliding window summaries for adjacent z distances.
    dist_csv = Path(args.dist_csv) if args.dist_csv else output_dir / "zt_zt_1.csv"
    dist_csv.parent.mkdir(parents=True, exist_ok=True)
    dist_fieldnames = [
            "task",
            "logdir",
            "ckpt",
            "episode",
            "window_start",
            "window_end",
            "boundary_rate",
            "read_prob_mean",
            "abs_l2_mean",
            "obs_l2_mean",
            "abs_stoch_l2_mean",
            "obs_stoch_l2_mean",
            "abs_l2_read_mean",
            "obs_l2_read_mean",
            "abs_stoch_l2_read_mean",
            "obs_stoch_l2_read_mean",
        ]
    dist_csv, write_header, mode = prepare_csv(dist_csv, dist_fieldnames, args.overwrite_csv)
    with dist_csv.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dist_fieldnames)
        if write_header:
            writer.writeheader()
        T = abs_l2.shape[1]
        win = args.dist_window
        stride = max(1, args.dist_stride)
        for start in range(0, max(1, T - win + 1), stride):
            end = min(T, start + win)
            sl = slice(start, end)
            bn = boundary_next[:, sl]
            rp = read_prob_next[:, sl] if read_prob_next is not None else None
            row = {
                "task": args.task,
                "logdir": str(logdir),
                "ckpt": str(ckpt_path),
                "episode": str(ep_path),
                "window_start": start,
                "window_end": end,
                "boundary_rate": float((bn > 0.5).mean()),
                "read_prob_mean": float(np.asarray(rp).mean()) if rp is not None else float("nan"),
                "abs_l2_mean": float(abs_l2[:, sl].mean()),
                "obs_l2_mean": float(obs_l2[:, sl].mean()),
                "abs_stoch_l2_mean": float(abs_stoch_l2[:, sl].mean()),
                "obs_stoch_l2_mean": float(obs_stoch_l2[:, sl].mean()),
                "abs_l2_read_mean": float(abs_l2[:, sl][bn > 0.5].mean()) if (bn > 0.5).any() else float("nan"),
                "obs_l2_read_mean": float(obs_l2[:, sl][bn > 0.5].mean()) if (bn > 0.5).any() else float("nan"),
                "abs_stoch_l2_read_mean": float(abs_stoch_l2[:, sl][bn > 0.5].mean()) if (bn > 0.5).any() else float("nan"),
                "obs_stoch_l2_read_mean": float(obs_stoch_l2[:, sl][bn > 0.5].mean()) if (bn > 0.5).any() else float("nan"),
            }
            writer.writerow(row)

    # Plot z_t to z_{t+1} distance summaries.
    plot_out = (
        Path(args.dist_plot_out)
        if args.dist_plot_out
        else output_dir / f"{dist_csv.stem}_plot.png"
    )
    window_start = []
    abs_l2_mean = []
    abs_stoch_l2_mean = []
    abs_l2_read = []
    obs_l2_mean = []
    obs_stoch_l2_mean = []
    obs_l2_read = []
    boundary_rate = []
    read_prob_mean = []
    with dist_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("task") != args.task:
                continue
            try:
                window_start.append(float(r.get("window_start", "nan")))
                abs_l2_mean.append(float(r.get("abs_l2_mean", "nan")))
                abs_l2_read.append(float(r.get("abs_l2_read_mean", "nan")))
                obs_l2_mean.append(float(r.get("obs_l2_mean", "nan")))
                obs_l2_read.append(float(r.get("obs_l2_read_mean", "nan")))
                abs_stoch_l2_mean.append(float(r.get("abs_stoch_l2_mean", "nan")))
                obs_stoch_l2_mean.append(float(r.get("obs_stoch_l2_mean", "nan")))
                boundary_rate.append(float(r.get("boundary_rate", "nan")))
                read_prob_mean.append(float(r.get("read_prob_mean", "nan")))
            except ValueError:
                continue
    if window_start:
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        axes[0].plot(window_start, abs_l2_mean, label="abs_mean_l2", color="#1f77b4")
        axes[0].plot(window_start, abs_stoch_l2_mean, label="abs_stoch_l2", color="#17becf", linestyle="--")
        axes[0].plot(window_start, abs_l2_read, label="abs_mean_l2@READ", color="#ff7f0e")
        axes[0].set_ylabel("Abs z distance")
        axes[0].legend(loc="upper right")

        axes[1].plot(window_start, obs_l2_mean, label="obs_mean_l2", color="#2ca02c")
        axes[1].plot(window_start, obs_stoch_l2_mean, label="obs_stoch_l2", color="#1f77b4", linestyle="--")
        axes[1].plot(window_start, obs_l2_read, label="obs_mean_l2@READ", color="#d62728")
        axes[1].set_ylabel("Obs s distance")
        axes[1].legend(loc="upper right")

        axes[2].plot(window_start, boundary_rate, label="boundary_rate", color="#9467bd")
        axes[2].set_ylabel("Boundary rate")
        axes[2].legend(loc="upper right")

        axes[3].plot(window_start, read_prob_mean, label="read_prob_mean", color="#8c564b")
        axes[3].set_ylabel("READ prob")
        axes[3].set_xlabel("Window start")
        axes[3].legend(loc="upper right")

        fig.suptitle(f"z_t to z_(t+1) distances - {args.task}")
        fig.tight_layout()
        fig.savefig(plot_out, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
