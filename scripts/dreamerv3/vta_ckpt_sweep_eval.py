#!/usr/bin/env python3
"""
Evaluate DreamerV3+VTA latent usage across checkpoints/logdirs.

This script is designed to answer:
  - Does VTA actually READ (boundary) or is it saturated to COPY?
  - Does the latent state used by the dynamics (abs_stoch) change across time?
  - How do these metrics evolve across checkpoints (if you keep multiple .pt files)?

It works with a single latest.pt per logdir, and will automatically pick up additional
checkpoint files if they exist (e.g., in logdir/checkpoints/).
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import gym
from torch import distributions as torchd
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src_dreamerv3 import models
from scripts.dreamerv3.vta_boundary_viz import load_config, pick_episode


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def adjacent_l2(x: np.ndarray) -> np.ndarray:
    diff = x[:, 1:, :] - x[:, :-1, :]
    return np.linalg.norm(diff, axis=-1)  # (B, T-1)


def prepare_csv(path: Path, fieldnames, overwrite: bool):
    """
    Returns (final_path, write_header, mode).
    If an existing file has a different header, default to writing a new *_v2.csv file,
    unless overwrite=True.
    """
    if overwrite:
        return path, True, "w"
    if not path.exists():
        return path, True, "w"
    try:
        with path.open() as f:
            header = f.readline().strip().split(",")
    except OSError:
        header = []
    if header == list(fieldnames):
        return path, False, "a"
    v2 = path.with_name(f"{path.stem}_v2{path.suffix}")
    return v2, True, "w"


def infer_task_from_config(logdir: Path) -> str | None:
    cfg = logdir / "config.yaml"
    if not cfg.exists():
        return None
    yaml = YAML(typ="safe", pure=True)
    try:
        data = yaml.load(cfg.read_text())
    except Exception:
        return None
    if isinstance(data, dict) and "task" in data:
        return str(data["task"])
    return None


def infer_task_from_name(logdir: Path) -> str | None:
    name = logdir.name
    m = re.match(r"^(atari_[a-z0-9_]+)_vta_", name)
    if m:
        return m.group(1)
    return None


def list_checkpoints(
    logdir: Path, include_latest: bool = True, latest_only: bool = False
) -> list[Path]:
    if latest_only:
        p = logdir / "latest.pt"
        return [p] if p.exists() else []

    cands: list[Path] = []
    if include_latest and (logdir / "latest.pt").exists():
        cands.append(logdir / "latest.pt")
    ckpt_dir = logdir / "checkpoints"
    if ckpt_dir.exists() and ckpt_dir.is_dir():
        cands.extend(sorted(ckpt_dir.glob("*.pt")))
    # Also include other .pt files at the logdir root (except latest.pt already added).
    cands.extend([p for p in logdir.glob("*.pt") if p.name != "latest.pt"])

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for p in cands:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)

    def step_key(p: Path):
        # Try to extract a trailing integer step (e.g., model-12345.pt, 12345.pt).
        m = re.search(r"(\d+)(?=\.pt$)", p.name)
        if m:
            return (0, int(m.group(1)))
        # Fall back to mtime for stable ordering.
        try:
            return (1, int(p.stat().st_mtime))
        except OSError:
            return (2, 0)

    return sorted(unique, key=step_key)


def parse_ckpt_step(path: Path) -> int | None:
    # Prefer explicit step-XXXXXXXXX.pt naming.
    m = re.search(r"step-(\d+)\.pt$", path.name)
    if m:
        return int(m.group(1))
    # Fallback: trailing integer before .pt (e.g. model-12345.pt, 12345.pt).
    m = re.search(r"(\d+)(?=\.pt$)", path.name)
    if m:
        return int(m.group(1))
    return None


def filter_checkpoints(
    ckpts: list[Path], eval_every_k: int | None, eval_steps_k: list[int] | None
) -> list[Path]:
    if eval_every_k is None and not eval_steps_k:
        return ckpts

    if eval_every_k is not None and eval_every_k <= 0:
        raise ValueError("--eval_every_k must be positive")

    target_steps = None
    if eval_steps_k:
        target_steps = {int(k) * 1000 for k in eval_steps_k}

    interval = int(eval_every_k) * 1000 if eval_every_k is not None else None
    kept = []
    for ckpt in ckpts:
        ckpt_step = parse_ckpt_step(ckpt)
        if ckpt_step is None:
            # With point-filtering enabled, skip non-step checkpoints (e.g., latest.pt).
            continue
        env_step = ckpt_step * 4
        if target_steps is not None and env_step not in target_steps:
            continue
        if interval is not None and (env_step % interval) != 0:
            continue
        kept.append(ckpt)
    return kept


@dataclass(frozen=True)
class EvalInputs:
    task: str
    logdir: Path
    ckpt_path: Path
    episode_path: Path
    seed: int
    dist_window: int
    dist_stride: int
    vta_boundary_force_scale: float
    device: str
    configs: list[str]
    overrides: list[str]
    boundary_source: str



def _get_vta_overrides(logdir: Path) -> list[str]:
    """
    Extract VTA-specific training parameters from config.yaml in logdir.
    This ensures evaluation uses the same boundary mode/params as training.
    """
    cfg_path = logdir / "config.yaml"
    if not cfg_path.exists():
        return []
    
    yaml = YAML(typ="unsafe", pure=True)
    try:
        data = yaml.load(cfg_path.read_text())
    except Exception:
        return []
        
    overrides = []
    # Keys to foster from training config
    keys = [
        "vta_train_boundary_mode",
        "vta_train_fixed_k",
        "vta_train_bernoulli_p",
        "vta_train_exp_lambda",
    ]
    
    if isinstance(data, dict):
        for k in keys:
            if k in data:
                overrides.append(f"--{k}")
                overrides.append(str(data[k]))
                
    return overrides


def _rollout_states(wm, proc, boundary_source: str):
    if boundary_source == "posterior":
        reward_t = proc.get("reward", None)
        return wm.dynamics.observe(proc["embed"], proc["action"], proc["is_first"], reward=reward_t)
    if boundary_source != "prior":
        raise ValueError(f"Unknown boundary_source: {boundary_source}")

    # Prior-boundary rollout: use obs_step with post_boundary_logit=None at every time step.
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    embed = swap(proc["embed"])
    action = swap(proc["action"])
    is_first = swap(proc["is_first"])
    batch_size, seq_len = proc["embed"].shape[:2]

    state = wm.dynamics.initial(batch_size)
    posts = {k: [] for k in state.keys()}
    priors = {k: [] for k in state.keys()}
    for t in range(seq_len):
        post, prior = wm.dynamics.obs_step(
            state, action[t], embed[t], is_first[t], post_boundary_logit=None
        )
        for k in state.keys():
            posts[k].append(post[k])
            priors[k].append(prior[k])
        state = post
    posts = {k: swap(torch.stack(v, dim=0)) for k, v in posts.items()}
    priors = {k: swap(torch.stack(v, dim=0)) for k, v in priors.items()}
    return posts, priors


def evaluate_one(inputs: EvalInputs):
    np.random.seed(inputs.seed)
    torch.manual_seed(inputs.seed)

    config = load_config(
        inputs.configs,
        [
            "--task",
            inputs.task,
            "--dynamics_type",
            "vta",
            "--device",
            inputs.device,
            "--vta_boundary_force_scale",
            str(inputs.vta_boundary_force_scale),
            *inputs.overrides,
        ],
    )

    with np.load(inputs.episode_path) as ep:
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

    wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(inputs.device)
    state = torch.load(inputs.ckpt_path, map_location=inputs.device)["agent_state_dict"]
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

    wm.eval()
    with torch.no_grad():
        proc = wm.preprocess(data)
        proc["embed"] = wm.encoder(proc)
        post, prior = _rollout_states(wm, proc, inputs.boundary_source)

        # For prior-boundary evaluation, use prior states for boundary/read masking
        # and abs_stoch distance, matching "prior-driven read" analysis.
        state_for_read = prior if inputs.boundary_source == "prior" else post
        state_for_abs_stoch = prior if inputs.boundary_source == "prior" else post

        abs_mean = post["abs_mean"].detach().cpu().numpy()
        obs_mean = post["obs_mean"].detach().cpu().numpy()
        abs_stoch = state_for_abs_stoch["abs_stoch"].detach().cpu().numpy()
        obs_stoch = post["obs_stoch"].detach().cpu().numpy()
        boundary = state_for_read["boundary"].squeeze(-1).detach().cpu().numpy()  # (B, T)
        boundary_logit = state_for_read.get("boundary_logit", None)
        read_prob = (
            torch.softmax(boundary_logit, dim=-1)[..., 0].detach().cpu().numpy()
            if boundary_logit is not None else None
        )
        post_abs = torchd.normal.Normal(post["abs_mean"], post["abs_std"])
        prior_abs = torchd.normal.Normal(prior["abs_mean"], prior["abs_std"])
        post_obs = torchd.normal.Normal(post["obs_mean"], post["obs_std"])
        prior_obs = torchd.normal.Normal(prior["obs_mean"], prior["obs_std"])
        abs_kl = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_abs, 1),
            torchd.independent.Independent(prior_abs, 1),
        ).detach().cpu().numpy()
        obs_kl = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_obs, 1),
            torchd.independent.Independent(prior_obs, 1),
        ).detach().cpu().numpy()

    abs_l2 = adjacent_l2(abs_mean)
    obs_l2 = adjacent_l2(obs_mean)
    abs_stoch_l2 = adjacent_l2(abs_stoch)
    obs_stoch_l2 = adjacent_l2(obs_stoch)

    boundary_next = boundary[:, 1:]
    read_prob_next = read_prob[:, 1:] if read_prob is not None else None

    summary = {
        "task": inputs.task,
        "logdir": str(inputs.logdir),
        "ckpt": str(inputs.ckpt_path),
        "episode": str(inputs.episode_path),
        "seed": int(inputs.seed),
        "dist_window": int(inputs.dist_window),
        "dist_stride": int(inputs.dist_stride),
        "vta_boundary_force_scale": float(inputs.vta_boundary_force_scale),
        "boundary_source": inputs.boundary_source,
        "boundary_rate": float((boundary > 0.5).mean()),
        "read_prob_mean": float(np.asarray(read_prob).mean()) if read_prob is not None else float("nan"),
        "read_prob_p99": _quantile(read_prob, 0.99) if read_prob is not None else float("nan"),
        "abs_kl_mean": float(abs_kl.mean()),
        "abs_kl_std": float(abs_kl.std()),
        "obs_kl_mean": float(obs_kl.mean()),
        "obs_kl_std": float(obs_kl.std()),
        "abs_l2_mean": float(abs_l2.mean()),
        "obs_l2_mean": float(obs_l2.mean()),
        "abs_stoch_l2_mean": float(abs_stoch_l2.mean()),
        "obs_stoch_l2_mean": float(obs_stoch_l2.mean()),
        "abs_l2_read_mean": float(abs_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
        "obs_l2_read_mean": float(obs_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
        "abs_stoch_l2_read_mean": float(abs_stoch_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
        "obs_stoch_l2_read_mean": float(obs_stoch_l2[boundary_next > 0.5].mean()) if (boundary_next > 0.5).any() else float("nan"),
    }

    # Sliding windows over time.
    T = abs_l2.shape[1]
    win = int(inputs.dist_window)
    stride = max(1, int(inputs.dist_stride))
    windows = []
    for start in range(0, max(1, T - win + 1), stride):
        end = min(T, start + win)
        sl = slice(start, end)
        bn = boundary_next[:, sl]
        rp = read_prob_next[:, sl] if read_prob_next is not None else None
        windows.append(
            {
                "task": inputs.task,
                "logdir": str(inputs.logdir),
                "ckpt": str(inputs.ckpt_path),
                "episode": str(inputs.episode_path),
                "seed": int(inputs.seed),
                "boundary_source": inputs.boundary_source,
                "window_start": int(start),
                "window_end": int(end),
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
        )

    return summary, windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdirs",
        nargs="+",
        required=True,
        help="One or more Dreamer logdirs. Each may contain latest.pt and optionally more checkpoints.",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks aligned to --logdirs.")
    parser.add_argument(
        "--episode",
        default=None,
        help="Explicit episode .npz path. If omitted, uses the longest episode in --episodes_dir for each logdir.",
    )
    parser.add_argument("--episodes_dir", default="train_eps")
    parser.add_argument("--output_dir", default="logs/vta_z_eval", help="Output directory (repo-relative).")
    parser.add_argument("--configs", nargs="+", default=["atari100k"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vta_boundary_force_scale", type=float, default=0.0)
    parser.add_argument(
        "--boundary_source",
        choices=["posterior", "prior"],
        default="posterior",
        help="Boundary source used during observe rollout.",
    )
    parser.add_argument("--dist_window", type=int, default=20)
    parser.add_argument("--dist_stride", type=int, default=5)
    parser.add_argument("--overwrite_csv", action="store_true")
    parser.add_argument(
        "--include_latest",
        action="store_true",
        help="Include latest.pt when other checkpoints exist (default: true).",
        default=True,
    )
    parser.add_argument(
        "--latest_only",
        action="store_true",
        help="Evaluate ONLY latest.pt, ignoring other checkpoints.",
    )
    parser.add_argument(
        "--eval_every_k",
        type=int,
        default=None,
        help="Evaluate only checkpoints at env-step multiples of this k interval (e.g., 100 -> 0k,100k,200k,...).",
    )
    parser.add_argument(
        "--eval_steps_k",
        nargs="+",
        type=int,
        default=None,
        help="Evaluate only these env-step points in k (e.g., 0 100 200 300 400).",
    )
    args, overrides = parser.parse_known_args()

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "vta_z_summary.csv"
    windows_csv = out_dir / "vta_z_windows.csv"

    summary_fieldnames = [
        "task",
        "logdir",
        "ckpt",
        "episode",
        "seed",
        "dist_window",
        "dist_stride",
        "vta_boundary_force_scale",
        "boundary_source",
        "boundary_rate",
        "read_prob_mean",
        "read_prob_p99",
        "abs_kl_mean",
        "abs_kl_std",
        "obs_kl_mean",
        "obs_kl_std",
        "abs_l2_mean",
        "obs_l2_mean",
        "abs_stoch_l2_mean",
        "obs_stoch_l2_mean",
        "abs_l2_read_mean",
        "obs_l2_read_mean",
        "abs_stoch_l2_read_mean",
        "obs_stoch_l2_read_mean",
    ]
    windows_fieldnames = [
        "task",
        "logdir",
        "ckpt",
        "episode",
        "seed",
        "boundary_source",
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

    summary_csv, summary_header, summary_mode = prepare_csv(
        summary_csv, summary_fieldnames, args.overwrite_csv
    )
    windows_csv, windows_header, windows_mode = prepare_csv(
        windows_csv, windows_fieldnames, args.overwrite_csv
    )

    logdirs = [Path(p) for p in args.logdirs]
    tasks = list(args.tasks) if args.tasks is not None else [None] * len(logdirs)
    if len(tasks) not in (0, len(logdirs)):
        raise ValueError("--tasks must have the same length as --logdirs (or be omitted).")

    with summary_csv.open(summary_mode, newline="") as sf, windows_csv.open(
        windows_mode, newline=""
    ) as wf:
        sw = csv.DictWriter(sf, fieldnames=summary_fieldnames)
        ww = csv.DictWriter(wf, fieldnames=windows_fieldnames)
        if summary_header:
            sw.writeheader()
        if windows_header:
            ww.writeheader()

        for logdir, task in zip(logdirs, tasks):
            inferred = task or infer_task_from_config(logdir) or infer_task_from_name(logdir)
            if not inferred:
                raise ValueError(
                    f"Could not infer task for {logdir}. Provide it via --tasks."
                )
            ckpts = list_checkpoints(
                logdir, include_latest=args.include_latest, latest_only=args.latest_only
            )
            ckpts = filter_checkpoints(
                ckpts, eval_every_k=args.eval_every_k, eval_steps_k=args.eval_steps_k
            )
            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoints matched under {logdir} "
                    f"(eval_every_k={args.eval_every_k}, eval_steps_k={args.eval_steps_k})"
                )

            logdir_overrides = _get_vta_overrides(logdir)
            # Combine CLI overrides (from parse_known_args) with logdir config overrides.
            # CLI overrides should take precedence, so put them last.
            final_overrides = logdir_overrides + list(overrides)

            ep = Path(args.episode) if args.episode else pick_episode(
                logdir / args.episodes_dir, None
            )
            for ckpt in ckpts:
                inputs = EvalInputs(
                    task=inferred,
                    logdir=logdir,
                    ckpt_path=ckpt,
                    episode_path=ep,
                    seed=args.seed,
                    dist_window=args.dist_window,
                    dist_stride=args.dist_stride,
                    vta_boundary_force_scale=args.vta_boundary_force_scale,
                    device=device,
                    configs=list(args.configs),
                    overrides=final_overrides,
                    boundary_source=args.boundary_source,
                )
                summary, windows = evaluate_one(inputs)
                sw.writerow(summary)
                for row in windows:
                    ww.writerow(row)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {windows_csv}")


if __name__ == "__main__":
    main()
