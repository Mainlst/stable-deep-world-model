#!/usr/bin/env python3
"""
Plot 3-seed metric trajectory at:
  - first available checkpoint step
  - then every 100k env steps (100k, 200k, ...)

Expected input CSV format is vta_z_summary.csv (from vta_ckpt_sweep_eval.py),
which includes at least:
  - logdir
  - ckpt
  - metric column (default: obs_stoch_l2_mean)
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", required=True, help="Path to vta_z_summary.csv")
    p.add_argument(
        "--metric",
        default="obs_stoch_l2_mean",
        help="Metric column to plot (default: obs_stoch_l2_mean)",
    )
    p.add_argument(
        "--max_env_step",
        type=int,
        default=400000,
        help="Maximum environment steps to include (default: 400000)",
    )
    p.add_argument(
        "--every_k",
        type=int,
        default=100,
        help="Interval in k-steps after the first point (default: 100)",
    )
    p.add_argument(
        "--out_png",
        default=None,
        help="Output figure path (default: <summary_dir>/<metric>_first_then_every100k_3seed.png)",
    )
    p.add_argument(
        "--out_csv",
        default=None,
        help="Output extracted table path (default: <summary_dir>/<metric>_first_then_every100k_3seed.csv)",
    )
    return p.parse_args()


def extract_rows(summary_csv: Path, metric: str):
    rows = []
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric not in row:
                raise KeyError(f"Metric column not found: {metric}")
            m_seed = re.search(r"seed(\d+)$", row["logdir"])
            m_step = re.search(r"step-(\d+)\.pt$", row["ckpt"])
            if not m_seed or not m_step:
                continue
            run_seed = int(m_seed.group(1))
            env_step = int(m_step.group(1)) * 4  # internal step -> env step
            try:
                value = float(row[metric])
            except ValueError:
                value = float("nan")
            rows.append((run_seed, env_step, value))
    return rows


def main():
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    rows = extract_rows(summary_csv, args.metric)
    if not rows:
        raise RuntimeError(f"No usable rows found in: {summary_csv}")

    first_env_step = min(env for _, env, _ in rows)
    interval = args.every_k * 1000
    targets = [first_env_step] + list(range(interval, args.max_env_step + 1, interval))
    targets = sorted(set(targets))

    data = {0: {}, 1: {}, 2: {}}
    for seed, env, value in rows:
        if seed in data and env in targets:
            data[seed][env] = value

    out_dir = summary_csv.parent
    out_png = (
        Path(args.out_png)
        if args.out_png
        else out_dir / f"{args.metric}_first_then_every{args.every_k}k_3seed.png"
    )
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else out_dir / f"{args.metric}_first_then_every{args.every_k}k_3seed.csv"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "env_step", args.metric])
        for seed in [0, 1, 2]:
            for x in targets:
                v = data[seed].get(x, float("nan"))
                w.writerow([seed, x, v])

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
    for seed in [0, 1, 2]:
        xs = targets
        ys = [data[seed].get(x, float("nan")) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=6.0, color=colors[seed], label=f"Seed {seed}")

    ax.set_xlim(min(targets), max(targets))
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{x//1000}k" for x in targets])
    ax.set_xlabel("Environment Steps (k)", fontsize=13)
    ax.set_ylabel(
        "Mean Obs-level Transition Distance\n$\\|\\mathbf{s}_{t+1}-\\mathbf{s}_t\\|_2$",
        fontsize=12,
    )
    ax.set_title("Training Dynamics of Observation-Level Latent Transition Distance", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout(pad=1.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
