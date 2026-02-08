#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_first_step(metrics_path: Path):
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        for line in f:
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in j:
                return int(j["step"])
    return None


def read_last_step(metrics_path: Path):
    if not metrics_path.exists():
        return None
    last = None
    with metrics_path.open() as f:
        for line in f:
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in j:
                last = int(j["step"])
    return last


def parse_task_from_stats(stats_path: Path):
    if not stats_path.exists():
        return None
    with stats_path.open() as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row:
        return row.get("task")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Run directory containing zcheck/")
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=4,
        help="Action repeat to convert internal steps to env steps (default: 4).",
    )
    parser.add_argument(
        "--out_png",
        default=None,
        help="Output PNG path (default: <run_dir>/zcheck_summary.png)",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path (default: <run_dir>/zcheck_summary.csv)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task name for the plot title (auto-detected if omitted).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    zcheck_dir = run_dir / "zcheck"
    if not zcheck_dir.exists():
        raise SystemExit(f"Missing zcheck dir: {zcheck_dir}")

    metrics_path = run_dir / "metrics.jsonl"
    start_step = read_first_step(metrics_path)
    last_step = read_last_step(metrics_path)

    rows = []
    task_name = args.task
    for sub in sorted(zcheck_dir.iterdir()):
        if not sub.is_dir():
            continue
        stats = sub / "boundary_stats.csv"
        if not stats.exists():
            continue
        with stats.open() as f:
            reader = csv.DictReader(f)
            data = next(reader, None)
        if not data:
            continue
        if task_name is None:
            task_name = data.get("task")

        name = sub.name
        step = None
        if name == "init_latest":
            step = start_step
        else:
            m = re.match(r"step-(\d+)", name)
            if m:
                step = int(m.group(1)) * args.action_repeat
            elif name == "latest":
                step = last_step

        def _f(x):
            return float("nan") if x == "nan" else float(x)

        rows.append(
            {
                "ckpt": name,
                "step": step,
                "abs_l2_mean": float(data["abs_l2_mean"]),
                "abs_stoch_l2_mean": float(data["abs_stoch_l2_mean"]),
                "abs_l2_read_mean": _f(data["abs_l2_read_mean"]),
                "abs_stoch_l2_read_mean": _f(data["abs_stoch_l2_read_mean"]),
                "boundary_rate": float(data["boundary_rate"]),
                "read_prob_mean": float(data["read_prob_mean"]),
            }
        )

    if not rows:
        raise SystemExit("No boundary_stats.csv found under zcheck/")

    rows.sort(key=lambda r: (r["step"] is None, r["step"] if r["step"] is not None else 0))

    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "zcheck_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    steps = np.array([r["step"] for r in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, [r["abs_l2_mean"] for r in rows], marker="o", label="abs_l2_mean")
    axes[0].plot(
        steps, [r["abs_stoch_l2_mean"] for r in rows], marker="o", label="abs_stoch_l2_mean"
    )
    axes[0].set_ylabel("||z_{t+1}-z_t||")
    axes[0].set_title("Latent Distance (all steps)")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        steps, [r["abs_l2_read_mean"] for r in rows], marker="o", label="abs_l2_read_mean"
    )
    axes[1].plot(
        steps,
        [r["abs_stoch_l2_read_mean"] for r in rows],
        marker="o",
        label="abs_stoch_l2_read_mean",
    )
    axes[1].set_ylabel("||z_{t+1}-z_t|| @READ")
    axes[1].set_title("Latent Distance (READ only)")
    axes[1].legend(loc="upper right")

    axes[2].plot(steps, [r["boundary_rate"] for r in rows], marker="o", label="boundary_rate")
    axes[2].plot(
        steps, [r["read_prob_mean"] for r in rows], marker="o", label="read_prob_mean"
    )
    axes[2].set_ylabel("Boundary / READ prob")
    axes[2].set_title("Boundary Usage")
    axes[2].set_xlabel("Env step")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    title = f"z(t)->z(t+1) summary"
    if task_name:
        title = f"{title} - {task_name}"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    out_png = Path(args.out_png) if args.out_png else run_dir / "zcheck_summary.png"
    fig.savefig(out_png, dpi=150)
    print(out_png)
    print(out_csv)


if __name__ == "__main__":
    main()
