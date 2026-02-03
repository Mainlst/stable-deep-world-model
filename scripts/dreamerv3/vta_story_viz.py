#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_font():
    preferred = [
        "Noto Sans CJK JP",
        "Noto Sans CJK",
        "Noto Serif CJK JP",
        "Noto Serif CJK",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return None


def moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def load_abs_ctx_series(path):
    by_task = {}
    if not Path(path).exists():
        return by_task
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task")
            if not task:
                continue
            try:
                t = int(row.get("t", 0))
                abs_kl = float(row.get("abs_kl_post_prior", "nan"))
                ctx_kl = float(row.get("ctx_kl", "nan"))
            except ValueError:
                continue
            by_task.setdefault(task, {"t": [], "abs_kl": [], "ctx_kl": []})
            by_task[task]["t"].append(t)
            by_task[task]["abs_kl"].append(abs_kl)
            by_task[task]["ctx_kl"].append(ctx_kl)
    for task, data in by_task.items():
        order = np.argsort(data["t"])
        by_task[task] = {
            "t": np.asarray(data["t"])[order],
            "abs_kl": np.asarray(data["abs_kl"])[order],
            "ctx_kl": np.asarray(data["ctx_kl"])[order],
        }
    return by_task


def load_zt_windows(paths):
    by_task = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row.get("task")
                if not task:
                    continue
                try:
                    w = float(row.get("window_start", "nan"))
                    abs_stoch = float(row.get("abs_stoch_l2_mean", "nan"))
                    obs_stoch = float(row.get("obs_stoch_l2_mean", "nan"))
                    boundary_rate = float(row.get("boundary_rate", "nan"))
                    read_prob = float(row.get("read_prob_mean", "nan"))
                except ValueError:
                    continue
                by_task.setdefault(
                    task,
                    {
                        "window_start": [],
                        "abs_stoch_l2_mean": [],
                        "obs_stoch_l2_mean": [],
                        "boundary_rate": [],
                        "read_prob_mean": [],
                    },
                )
                by_task[task]["window_start"].append(w)
                by_task[task]["abs_stoch_l2_mean"].append(abs_stoch)
                by_task[task]["obs_stoch_l2_mean"].append(obs_stoch)
                by_task[task]["boundary_rate"].append(boundary_rate)
                by_task[task]["read_prob_mean"].append(read_prob)
    for task, data in by_task.items():
        order = np.argsort(data["window_start"])
        by_task[task] = {
            "window_start": np.asarray(data["window_start"])[order],
            "abs_stoch_l2_mean": np.asarray(data["abs_stoch_l2_mean"])[order],
            "obs_stoch_l2_mean": np.asarray(data["obs_stoch_l2_mean"])[order],
            "boundary_rate": np.asarray(data["boundary_rate"])[order],
            "read_prob_mean": np.asarray(data["read_prob_mean"])[order],
        }
    return by_task


def plot_story(task, abs_ctx, zt, out_path, smooth):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)

    # 1) Latent prediction stability (latent step distance)
    if zt is not None and len(zt["window_start"]) > 0:
        x = zt["window_start"]
        z = zt["abs_stoch_l2_mean"]
        z_sm = moving_average(np.nan_to_num(z, nan=0.0), smooth)
        axes[0].plot(x, z_sm, color="#1f77b4", linewidth=2.0, label="z (stoch) Δt")
        axes[0].plot(x, z, color="#1f77b4", alpha=0.25, linewidth=1.0)
        axes[0].set_ylabel("Latent distance (||z_{t+1}-z_t||)")
        axes[0].set_title("Latent Prediction Stability")
        axes[0].legend(loc="upper right")
    else:
        axes[0].text(0.5, 0.5, "No z_t→z_{t+1} window data.", ha="center")
        axes[0].axis("off")

    # 2) Temporal abstraction / boundary usage
    if zt is not None and len(zt["window_start"]) > 0:
        x = zt["window_start"]
        b = zt["boundary_rate"]
        r = zt["read_prob_mean"]
        b_sm = moving_average(np.nan_to_num(b, nan=0.0), smooth)
        r_sm = moving_average(np.nan_to_num(r, nan=0.0), smooth)
        axes[1].plot(x, b_sm, color="#9467bd", linewidth=2.0, label="boundary_rate")
        axes[1].plot(x, r_sm, color="#8c564b", linewidth=2.0, label="read_prob_mean")
        axes[1].plot(x, b, color="#9467bd", alpha=0.2, linewidth=1.0)
        axes[1].plot(x, r, color="#8c564b", alpha=0.2, linewidth=1.0)
        axes[1].set_ylabel("Boundary / READ probability")
        axes[1].set_title("Temporal Abstraction (Boundary Usage)")
        axes[1].legend(loc="upper right")
    else:
        axes[1].text(0.5, 0.5, "No VTA boundary data.", ha="center")
        axes[1].axis("off")

    # 3) Prior vs posterior KL (prediction vs observation)
    if abs_ctx is not None and len(abs_ctx["t"]) > 0:
        t = abs_ctx["t"]
        kl = abs_ctx["abs_kl"]
        ctx = abs_ctx["ctx_kl"]
        kl_sm = moving_average(np.nan_to_num(kl, nan=0.0), smooth)
        ctx_sm = moving_average(np.nan_to_num(ctx, nan=0.0), smooth)
        axes[2].plot(t, kl_sm, color="#2ca02c", linewidth=2.0, label="KL(post_z || prior_z)")
        axes[2].plot(t, kl, color="#2ca02c", alpha=0.25, linewidth=1.0)
        axes[2].plot(t, ctx_sm, color="#d62728", linewidth=2.0, label="Context KL")
        axes[2].plot(t, ctx, color="#d62728", alpha=0.25, linewidth=1.0)
        axes[2].set_ylabel("KL divergence")
        axes[2].set_xlabel("Time Step")
        axes[2].set_title("Prior–Posterior Consistency (KL)")
        axes[2].legend(loc="upper right")
    else:
        axes[2].text(0.5, 0.5, "No KL series data.", ha="center")
        axes[2].axis("off")

    fig.suptitle(f"VTA Diagnostics - {task}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abs_ctx_csv",
        default="logs/vta_abs_ctx_kl_series.csv",
        help="CSV generated by vta_abs_kl_plot.py",
    )
    parser.add_argument(
        "--zt_csvs",
        nargs="+",
        default=["logs/zt_zt_1/*/zt_zt_1.csv"],
        help="One or more zt_zt_1.csv paths (globs allowed).",
    )
    parser.add_argument(
        "--out_dir",
        default="logs/vta_story_viz",
        help="Output directory for story plots.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task filter (e.g., atari_krull atari_frostbite).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Moving average window for visualization.",
    )
    args = parser.parse_args()

    configure_font()

    abs_ctx = load_abs_ctx_series(args.abs_ctx_csv)

    zt_paths = []
    for entry in args.zt_csvs:
        if "*" in entry or "?" in entry or "[" in entry:
            zt_paths.extend(sorted(Path().glob(entry)))
        else:
            zt_paths.append(Path(entry))
    zt_paths = [str(p) for p in zt_paths]
    zt = load_zt_windows(zt_paths)

    tasks = args.tasks or sorted(set(abs_ctx.keys()) | set(zt.keys()))
    if not tasks:
        raise SystemExit("No tasks found in the provided CSV files.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        plot_story(
            task,
            abs_ctx.get(task),
            zt.get(task),
            out_dir / f"{task}_story.png",
            args.smooth,
        )


if __name__ == "__main__":
    main()
