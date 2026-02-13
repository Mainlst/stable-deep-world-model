#!/usr/bin/env python3
"""
Plot 3-seed curve of abs_stoch_l2_read_mean with optional 0k (random-init) points.

Inputs:
  - vta_z_summary.csv (checkpoint sweep result)
  - optional per-seed boundary_stats.csv from random-init evaluation
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_seed(logdir: str) -> int | None:
    m = re.search(r"seed(\d+)", logdir)
    return int(m.group(1)) if m else None


def parse_step_from_ckpt(ckpt: str) -> int | None:
    m = re.search(r"step-(\d+)\.pt$", ckpt)
    return int(m.group(1)) if m else None


def load_summary(summary_csv: Path, metric: str):
    by_seed: dict[int, list[tuple[int, float]]] = {}
    rows = []
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {summary_csv}")
        for row in reader:
            seed = parse_seed(row["logdir"])
            ckpt_step = parse_step_from_ckpt(row["ckpt"])
            if seed is None or ckpt_step is None:
                continue
            val = float(row[metric])
            env_step = ckpt_step * 4
            by_seed.setdefault(seed, []).append((env_step, val))
            rows.append(row)
    return by_seed, rows, fieldnames


def filter_points_by_every_k(
    by_seed: dict[int, list[tuple[int, float]]], every_k: int | None
) -> dict[int, list[tuple[int, float]]]:
    if every_k is None:
        return by_seed
    interval = every_k * 1000
    if interval <= 0:
        raise ValueError("--every_k must be positive")
    filtered: dict[int, list[tuple[int, float]]] = {}
    for seed, pts in by_seed.items():
        filtered[seed] = [(x, y) for x, y in pts if x % interval == 0]
    return filtered


def dedupe_points_by_step(
    by_seed: dict[int, list[tuple[int, float]]],
) -> dict[int, list[tuple[int, float]]]:
    deduped: dict[int, list[tuple[int, float]]] = {}
    for seed, pts in by_seed.items():
        last_by_step: dict[int, float] = {}
        for x, y in pts:
            last_by_step[x] = y
        deduped[seed] = sorted(last_by_step.items(), key=lambda p: p[0])
    return deduped


def load_random_init_point(base_dir: Path, seed: int, metric: str) -> tuple[float, dict[str, str]]:
    stats_csv = (
        base_dir
        / f"atari_krull_seed{seed}"
        / "zcheck_random_init_evaleps_match_sweep"
        / "boundary_stats.csv"
    )
    if not stats_csv.exists():
        raise FileNotFoundError(stats_csv)
    with stats_csv.open() as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return float(row[metric]), row


def add_0k_rows(rows, fieldnames, seed, row_from_boundary_stats):
    newrow = {k: "" for k in fieldnames}
    newrow["task"] = row_from_boundary_stats.get("task", "atari_krull")
    newrow["logdir"] = row_from_boundary_stats["logdir"]
    newrow["ckpt"] = "step-000000000.pt"
    newrow["episode"] = row_from_boundary_stats["episode"]
    newrow["seed"] = str(seed)
    newrow["dist_window"] = "20"
    newrow["dist_stride"] = "5"
    newrow["vta_boundary_force_scale"] = "0.0"
    keys = [
        "boundary_rate",
        "read_prob_mean",
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
    for key in keys:
        if key in newrow and key in row_from_boundary_stats:
            newrow[key] = row_from_boundary_stats[key]
    rows.append(newrow)


def write_extended_csv(out_csv: Path, rows, fieldnames):
    def sort_key(row):
        seed = parse_seed(row["logdir"])
        step = parse_step_from_ckpt(row["ckpt"])
        return (seed if seed is not None else 10**9, step if step is not None else 10**9)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=sort_key)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)


def plot_curve(
    out_png: Path,
    by_seed: dict[int, list[tuple[int, float]]],
    metric: str,
    xtick_k: int = 50,
):
    colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
    plt.figure(figsize=(10.8, 6.0))
    for seed in sorted(by_seed):
        pts = sorted(by_seed[seed], key=lambda x: x[0])
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            color=colors.get(seed),
            label=f"seed {seed}",
        )

    plt.grid(True, alpha=0.25, linestyle="--")
    plt.xlim(0, 400000)
    ticks = np.arange(0, 400001, xtick_k * 1000)
    plt.xticks(ticks, [f"{int(t / 1000)}k" for t in ticks])
    plt.xlabel("Environment steps")
    plt.ylabel(f"{metric} (||z_t-z_(t-1)||_2 @ b_t=1)")
    plt.title("VTA read-step latent distance across 3 seeds")
    plt.legend(loc="best")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("logs/models/from_scratch_atari_krull_vta_3seed_to_400000_20260208_112207"),
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Defaults to {base_dir}/zt_zt_1_sweep_evaleps_unconstrained_3seed/vta_z_summary.csv",
    )
    parser.add_argument("--metric", default="abs_stoch_l2_read_mean")
    parser.add_argument("--include_0k", action="store_true")
    parser.add_argument(
        "--every_k",
        type=int,
        default=None,
        help="Keep only points at multiples of this interval in k-steps (e.g., 100 -> 0k,100k,...).",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=None,
        help="Defaults to {base_dir}/zt_zt_1_sweep_evaleps_unconstrained_3seed/z_distance_3seed_abs_stoch_l2_read_line_with0k.png",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="When include_0k, write merged CSV. Defaults to .../vta_z_summary_with0k.csv",
    )
    args = parser.parse_args()

    summary_csv = args.summary_csv or (
        args.base_dir / "zt_zt_1_sweep_evaleps_unconstrained_3seed" / "vta_z_summary.csv"
    )
    output_png = args.output_png or (
        args.base_dir
        / "zt_zt_1_sweep_evaleps_unconstrained_3seed"
        / ("z_distance_3seed_abs_stoch_l2_read_line_with0k.png" if args.include_0k else "z_distance_3seed_abs_stoch_l2_read_line.png")
    )
    output_csv = args.output_csv or (
        args.base_dir / "zt_zt_1_sweep_evaleps_unconstrained_3seed" / "vta_z_summary_with0k.csv"
    )

    by_seed, rows, fieldnames = load_summary(summary_csv, args.metric)
    if args.include_0k:
        for seed in sorted(by_seed):
            if any(x == 0 for x, _ in by_seed[seed]):
                continue
            val0, row0 = load_random_init_point(args.base_dir, seed, args.metric)
            by_seed[seed].append((0, val0))
            add_0k_rows(rows, fieldnames, seed, row0)
        write_extended_csv(output_csv, rows, fieldnames)
        print(f"Wrote merged CSV: {output_csv}")

    by_seed = filter_points_by_every_k(by_seed, args.every_k)
    by_seed = dedupe_points_by_step(by_seed)
    plot_curve(output_png, by_seed, args.metric, xtick_k=(args.every_k or 50))
    print(f"Wrote plot: {output_png}")


if __name__ == "__main__":
    main()
