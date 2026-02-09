#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src_dreamerv3 import models
from scripts.dreamerv3.vta_boundary_viz import load_config


def list_checkpoints(logdir: Path):
    ckpt_dir = logdir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("step-*.pt"))
    latest = logdir / "latest.pt"
    if latest.exists():
        ckpts.append(latest)
    return ckpts


def ckpt_env_step(ckpt: Path, action_repeat: int):
    if ckpt.name == "latest.pt":
        return None
    m = re.match(r"step-(\d+)\.pt", ckpt.name)
    if not m:
        return None
    return int(m.group(1)) * action_repeat


def safe_stats(x: np.ndarray):
    if x.size == 0:
        return dict(mean=np.nan, std=np.nan, median=np.nan, p90=np.nan, p99=np.nan)
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        median=float(np.median(x)),
        p90=float(np.quantile(x, 0.9)),
        p99=float(np.quantile(x, 0.99)),
    )


def make_world_model(config, images, actions, device):
    action_dim = actions.shape[-1]
    config.num_actions = action_dim
    obs_space = gym.spaces.Dict(
        {"image": gym.spaces.Box(0, 255, shape=images.shape[1:], dtype=np.uint8)}
    )
    act_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
    act_space.discrete = True
    return models.WorldModel(obs_space, act_space, step=0, config=config).to(device)


def eval_one(ckpt_path: Path, task: str, episode_path: Path, args):
    config = load_config(
        args.configs,
        [
            "--task",
            task,
            "--dynamics_type",
            "vta",
            "--device",
            args.device,
            "--vta_boundary_force_scale",
            str(args.vta_boundary_force_scale),
            "--vta_max_seg_len",
            str(args.vta_max_seg_len),
            "--vta_max_seg_num",
            str(args.vta_max_seg_num),
        ],
    )

    with np.load(episode_path) as ep:
        images = ep["image"]
        actions = ep["action"]
        reward = ep["reward"] if "reward" in ep else np.zeros((images.shape[0],), dtype=np.float32)
        is_first = ep["is_first"]
        is_terminal = ep["is_terminal"]

    wm = make_world_model(config, images, actions, args.device)
    state = torch.load(ckpt_path, map_location=args.device)["agent_state_dict"]
    wm_state = {k[len("_wm."):]: v for k, v in state.items() if k.startswith("_wm.")}
    wm_state = {k.replace("_orig_mod.", ""): v for k, v in wm_state.items()}
    wm.load_state_dict(wm_state, strict=True)
    wm.eval()

    data = {
        "image": images[None],
        "action": actions[None],
        "is_first": is_first[None],
        "is_terminal": is_terminal[None],
        "reward": reward[None],
    }

    with torch.no_grad():
        proc = wm.preprocess(data)
        embed = wm.encoder(proc)
        post, prior = wm.dynamics.observe(embed, proc["action"], proc["is_first"], reward=proc["reward"])

        boundary = post["boundary"].squeeze(-1).detach().cpu().numpy()[0]  # (T,)
        abs_stoch = post["abs_stoch"].detach().cpu().numpy()[0]  # (T, D)
        abs_kl = torch.distributions.kl.kl_divergence(
            torch.distributions.Independent(
                torch.distributions.Normal(post["abs_mean"], post["abs_std"]), 1
            ),
            torch.distributions.Independent(
                torch.distributions.Normal(prior["abs_mean"], prior["abs_std"]), 1
            ),
        ).detach().cpu().numpy()[0]
        obs_kl = torch.distributions.kl.kl_divergence(
            torch.distributions.Independent(
                torch.distributions.Normal(post["obs_mean"], post["obs_std"]), 1
            ),
            torch.distributions.Independent(
                torch.distributions.Normal(prior["obs_mean"], prior["obs_std"]), 1
            ),
        ).detach().cpu().numpy()[0]
        abs_pred_err = np.linalg.norm(
            post["abs_mean"].detach().cpu().numpy()[0] - prior["abs_mean"].detach().cpu().numpy()[0],
            axis=-1,
        )
        obs_pred_err = np.linalg.norm(
            post["obs_mean"].detach().cpu().numpy()[0] - prior["obs_mean"].detach().cpu().numpy()[0],
            axis=-1,
        )

    # A) COPY/UPDATE split using m_{t-1} for ||z_{t+1} - z_t||.
    dz = np.linalg.norm(abs_stoch[1:] - abs_stoch[:-1], axis=-1)  # t=0..T-2
    m_prev = boundary[:-1] > 0.5
    dz_copy = dz[~m_prev]
    dz_update = dz[m_prev]
    copy_stats = safe_stats(dz_copy)
    update_stats = safe_stats(dz_update)
    two_peak_ratio = (
        float(update_stats["mean"] / copy_stats["mean"])
        if np.isfinite(update_stats["mean"]) and np.isfinite(copy_stats["mean"]) and copy_stats["mean"] > 0
        else np.nan
    )

    # B) Segment length distribution and event synchronization.
    T = boundary.shape[0]
    update_idx = np.where(boundary > 0.5)[0]
    starts = [0] + [int(i) for i in update_idx if int(i) > 0]
    starts = sorted(set(starts))
    ends = starts[1:] + [T]
    seg_lengths = np.asarray([e - s for s, e in zip(starts, ends)], dtype=np.int32)
    seg_stats = safe_stats(seg_lengths.astype(np.float32))

    reward_ev = np.abs(reward) > args.reward_event_eps
    term_ev = is_terminal > 0.5
    boundary_ev = boundary > 0.5

    def near_boundary(mask, radius):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return np.nan
        out = []
        for i in idx:
            lo = max(0, i - radius)
            hi = min(T, i + radius + 1)
            out.append(np.any(boundary_ev[lo:hi]))
        return float(np.mean(out))

    boundary_at_reward = float(np.mean(boundary_ev[reward_ev])) if np.any(reward_ev) else np.nan
    reward_near_boundary = near_boundary(reward_ev, args.sync_radius)
    boundary_at_terminal = float(np.mean(boundary_ev[term_ev])) if np.any(term_ev) else np.nan
    terminal_near_boundary = near_boundary(term_ev, args.sync_radius)

    # C) Boundary-centered information curves.
    metrics = {
        "abs_kl": abs_kl,
        "obs_kl": obs_kl,
        "abs_pred_err": abs_pred_err,
        "obs_pred_err": obs_pred_err,
    }
    offsets = np.arange(-args.boundary_window, args.boundary_window + 1, dtype=np.int32)
    bidx = np.where(boundary_ev)[0]
    curve_rows = []
    c_summary = {}
    for name, series in metrics.items():
        vals = []
        for off in offsets:
            xs = []
            for b in bidx:
                i = b + int(off)
                if 0 <= i < T:
                    xs.append(series[i])
            vals.append(float(np.mean(xs)) if xs else np.nan)
            curve_rows.append((name, int(off), vals[-1]))
        vals_arr = np.asarray(vals, dtype=np.float32)
        pre = vals_arr[offsets < 0]
        post = vals_arr[offsets > 0]
        pre_mean = float(np.nanmean(pre)) if np.any(np.isfinite(pre)) else np.nan
        post_mean = float(np.nanmean(post)) if np.any(np.isfinite(post)) else np.nan
        c_summary[f"{name}_pre_mean"] = pre_mean
        c_summary[f"{name}_post_mean"] = post_mean
        c_summary[f"{name}_post_minus_pre"] = (
            float(post_mean - pre_mean) if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan
        )

    summary = {
        "ckpt": ckpt_path.name,
        "step_env": ckpt_env_step(ckpt_path, args.action_repeat),
        "episode": str(episode_path),
        "T": int(T),
        "boundary_rate": float(np.mean(boundary_ev)),
        "update_count": int(np.sum(boundary_ev)),
        "A_copy_mean": copy_stats["mean"],
        "A_copy_std": copy_stats["std"],
        "A_copy_p99": copy_stats["p99"],
        "A_update_mean": update_stats["mean"],
        "A_update_std": update_stats["std"],
        "A_update_p99": update_stats["p99"],
        "A_update_over_copy_mean": two_peak_ratio,
        "B_seg_mean": seg_stats["mean"],
        "B_seg_median": seg_stats["median"],
        "B_seg_p90": seg_stats["p90"],
        "B_seg_count": int(seg_lengths.size),
        "B_boundary_at_reward": boundary_at_reward,
        "B_reward_near_boundary": reward_near_boundary,
        "B_boundary_at_terminal": boundary_at_terminal,
        "B_terminal_near_boundary": terminal_near_boundary,
        **c_summary,
    }
    return summary, seg_lengths, curve_rows, dz_copy, dz_update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--configs", nargs="+", default=["atari100k"])
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--vta_boundary_force_scale", type=float, default=0.0)
    parser.add_argument("--vta_max_seg_len", type=int, default=1_000_000)
    parser.add_argument("--vta_max_seg_num", type=int, default=1_000_000)
    parser.add_argument("--reward_event_eps", type=float, default=1e-8)
    parser.add_argument("--sync_radius", type=int, default=5)
    parser.add_argument("--boundary_window", type=int, default=10)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    episode = Path(args.episode)
    ckpts = list_checkpoints(logdir)
    if not ckpts:
        raise SystemExit(f"No checkpoints under {logdir}")

    summaries = []
    seg_rows = []
    curve_rows_all = []
    latest_copy = latest_update = None
    for ckpt in ckpts:
        print(f"[eval] {ckpt.name}")
        summary, seg_lengths, curve_rows, dz_copy, dz_update = eval_one(ckpt, args.task, episode, args)
        summaries.append(summary)
        for i, l in enumerate(seg_lengths):
            seg_rows.append({"ckpt": ckpt.name, "segment_index": i, "segment_len": int(l)})
        for metric, off, v in curve_rows:
            curve_rows_all.append({"ckpt": ckpt.name, "metric": metric, "offset": off, "value": v})
        if ckpt.name == "latest.pt":
            latest_copy = dz_copy
            latest_update = dz_update

    def sort_key(r):
        s = r.get("step_env")
        return (1e18 if s is None else s, r["ckpt"])

    summaries.sort(key=sort_key)
    summary_csv = out / "additional_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    seg_csv = out / "segment_lengths.csv"
    with seg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ckpt", "segment_index", "segment_len"])
        writer.writeheader()
        writer.writerows(seg_rows)

    curve_csv = out / "boundary_aligned_curves.csv"
    with curve_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ckpt", "metric", "offset", "value"])
        writer.writeheader()
        writer.writerows(curve_rows_all)

    # Plots
    xs = np.array([np.nan if r["step_env"] is None else float(r["step_env"]) for r in summaries], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(xs, [r["A_copy_mean"] for r in summaries], marker="o", label="COPY mean")
    axes[0].plot(xs, [r["A_update_mean"] for r in summaries], marker="o", label="UPDATE mean")
    axes[0].set_ylabel("||z(t+1)-z(t)||")
    axes[0].set_title("A: COPY vs UPDATE")
    axes[0].legend(loc="upper left")

    axes[1].plot(xs, [r["B_seg_mean"] for r in summaries], marker="o", label="seg mean")
    axes[1].plot(xs, [r["B_seg_median"] for r in summaries], marker="o", label="seg median")
    axes[1].set_ylabel("segment length")
    axes[1].set_title("B: Segment Length")
    axes[1].legend(loc="upper left")

    axes[2].plot(xs, [r["abs_kl_post_minus_pre"] for r in summaries], marker="o", label="abs_kl post-pre")
    axes[2].plot(xs, [r["obs_kl_post_minus_pre"] for r in summaries], marker="o", label="obs_kl post-pre")
    axes[2].plot(xs, [r["abs_pred_err_post_minus_pre"] for r in summaries], marker="o", label="abs_pred_err post-pre")
    axes[2].plot(xs, [r["obs_pred_err_post_minus_pre"] for r in summaries], marker="o", label="obs_pred_err post-pre")
    axes[2].set_ylabel("delta")
    axes[2].set_xlabel("env step")
    axes[2].set_title("C: Boundary-centered post - pre")
    axes[2].legend(loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "additional_overview.png", dpi=160)
    plt.close(fig)

    # Latest checkpoint detailed hist.
    if latest_copy is not None and latest_update is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        bins = np.linspace(0, np.nanpercentile(np.concatenate([latest_copy, latest_update]), 99.5), 120)
        ax.hist(latest_copy, bins=bins, alpha=0.7, density=True, label="COPY")
        ax.hist(latest_update, bins=bins, alpha=0.7, density=True, label="UPDATE")
        ax.set_title("A detailed (latest): COPY vs UPDATE distance")
        ax.set_xlabel("||z(t+1)-z(t)||")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "A_latest_hist.png", dpi=180)
        plt.close(fig)

    # Latest checkpoint boundary-aligned curves.
    latest_name = "latest.pt"
    rows = [r for r in curve_rows_all if r["ckpt"] == latest_name]
    if rows:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for metric in ("abs_kl", "obs_kl"):
            mrows = [r for r in rows if r["metric"] == metric]
            mrows.sort(key=lambda x: x["offset"])
            axes[0].plot([r["offset"] for r in mrows], [r["value"] for r in mrows], marker="o", label=metric)
        axes[0].axvline(0, color="k", linestyle="--", alpha=0.5)
        axes[0].set_title("C detailed (latest): KL around boundary")
        axes[0].set_ylabel("KL")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        for metric in ("abs_pred_err", "obs_pred_err"):
            mrows = [r for r in rows if r["metric"] == metric]
            mrows.sort(key=lambda x: x["offset"])
            axes[1].plot([r["offset"] for r in mrows], [r["value"] for r in mrows], marker="o", label=metric)
        axes[1].axvline(0, color="k", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("prediction error")
        axes[1].set_xlabel("offset from boundary")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "C_latest_curves.png", dpi=180)
        plt.close(fig)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {seg_csv}")
    print(f"Wrote: {curve_csv}")
    print(f"Wrote: {out / 'additional_overview.png'}")


if __name__ == "__main__":
    main()
