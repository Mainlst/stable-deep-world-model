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
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src_dreamerv3 import models
from scripts.dreamerv3.vta_boundary_viz import load_config, pick_episode, compute_vta_stats


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


def list_checkpoints(logdir: Path, include_latest: bool = True) -> list[Path]:
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

    stats = compute_vta_stats(wm, data)

    wm.eval()
    with torch.no_grad():
        proc = wm.preprocess(data)
        embed = wm.encoder(proc)
        reward_t = proc.get("reward", None)
        post, _ = wm.dynamics.observe(embed, proc["action"], proc["is_first"], reward=reward_t)

        abs_mean = post["abs_mean"].detach().cpu().numpy()
        obs_mean = post["obs_mean"].detach().cpu().numpy()
        abs_stoch = post["abs_stoch"].detach().cpu().numpy()
        obs_stoch = post["obs_stoch"].detach().cpu().numpy()

        boundary = stats["boundary"]  # (B, T)
        boundary_logit = post.get("boundary_logit", None)
        read_prob = (
            torch.softmax(boundary_logit, dim=-1)[..., 0].detach().cpu().numpy()
            if boundary_logit is not None
            else stats.get("read_prob", None)
        )

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
        "boundary_rate": float((boundary > 0.5).mean()),
        "read_prob_mean": float(np.asarray(read_prob).mean()) if read_prob is not None else float("nan"),
        "read_prob_p99": _quantile(read_prob, 0.99) if read_prob is not None else float("nan"),
        "abs_kl_mean": float(stats["abs_kl"].mean()),
        "abs_kl_std": float(stats["abs_kl"].std()),
        "obs_kl_mean": float(stats["obs_kl"].mean()),
        "obs_kl_std": float(stats["obs_kl"].std()),
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
    parser.add_argument("--dist_window", type=int, default=20)
    parser.add_argument("--dist_stride", type=int, default=5)
    parser.add_argument("--overwrite_csv", action="store_true")
    parser.add_argument(
        "--include_latest",
        action="store_true",
        help="Include latest.pt when other checkpoints exist (default: true).",
        default=True,
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
            ckpts = list_checkpoints(logdir, include_latest=args.include_latest)
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found under {logdir}")

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
                    overrides=list(overrides),
                )
                summary, windows = evaluate_one(inputs)
                sw.writerow(summary)
                for row in windows:
                    ww.writerow(row)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {windows_csv}")


if __name__ == "__main__":
    main()

