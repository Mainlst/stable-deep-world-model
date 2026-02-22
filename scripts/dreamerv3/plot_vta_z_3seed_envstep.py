#!/usr/bin/env python3
"""
Plot DreamerV3+VTA z-distance curves across 3 seeds on env-step axis.

Inputs:
  - vta_z_summary.csv
  - vta_z_windows.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_seed(logdir: str) -> int:
    m = re.search(r"seed(\d+)", logdir)
    return int(m.group(1)) if m else -1


def _parse_ckpt_step(ckpt: str) -> int:
    m = re.search(r"step-(\d+)\.pt$", ckpt)
    if m:
        return int(m.group(1))
    if ckpt.endswith("latest.pt"):
        # In this run, latest corresponds to the final 100k agent steps (=400k env steps).
        return 100000
    return -1


def _to_float(x: str) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def load_summary(summary_csv: Path, metric: str) -> dict[int, list[tuple[int, float]]]:
    by_key: dict[tuple[int, int], tuple[bool, float]] = {}
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = _parse_seed(row["logdir"])
            ckpt_step = _parse_ckpt_step(row["ckpt"])
            if seed < 0 or ckpt_step < 0:
                continue
            val = _to_float(row.get(metric, ""))
            if val is None:
                continue
            is_latest = row["ckpt"].endswith("latest.pt")
            key = (seed, ckpt_step)
            # Prefer explicit checkpoint file over latest for duplicated step.
            if key not in by_key:
                by_key[key] = (is_latest, val)
            else:
                old_latest, _ = by_key[key]
                if old_latest and not is_latest:
                    by_key[key] = (is_latest, val)

    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (seed, ckpt_step), (_, val) in by_key.items():
        env_step = ckpt_step * 4
        out[seed].append((env_step, val))
    for seed in out:
        out[seed].sort(key=lambda x: x[0])
    return out


def load_windows(windows_csv: Path, metric: str) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    accum: dict[tuple[int, int], list[float]] = defaultdict(list)
    with windows_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = _parse_seed(row["logdir"])
            ckpt_step = _parse_ckpt_step(row["ckpt"])
            if seed < 0 or ckpt_step < 0:
                continue
            val = _to_float(row.get(metric, ""))
            if val is None:
                continue
            accum[(seed, ckpt_step * 4)].append(val)

    out: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    for (seed, env_step), values in accum.items():
        arr = np.asarray(values, dtype=float)
        out[seed][env_step] = (
            float(np.quantile(arr, 0.10)),
            float(np.mean(arr)),
            float(np.quantile(arr, 0.50)),
            float(np.quantile(arr, 0.90)),
        )
    return out


def _style_axis(ax):
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 400000)
    ticks = np.arange(0, 400001, 50000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t/1000)}k" for t in ticks])


def plot_full(
    out_png: Path,
    summary: dict[int, list[tuple[int, float]]],
    metric_label: str,
    colors: dict[int, str],
):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for seed in sorted(summary):
        pts = summary[seed]
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=5, label=f"seed {seed}", color=colors.get(seed))

    _style_axis(ax)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric_label)
    ax.set_title("DreamerV3+VTA: Adjacent z distance across training (full scale)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_robust(
    out_png: Path,
    summary: dict[int, list[tuple[int, float]]],
    metric_label: str,
    colors: dict[int, str],
):
    all_vals = np.asarray([y for pts in summary.values() for _, y in pts], dtype=float)
    q95 = float(np.quantile(all_vals, 0.95))
    q99 = float(np.quantile(all_vals, 0.99))
    if q99 > q95 * 3.0:
        y_top = q95 * 1.25
    else:
        y_top = float(np.max(all_vals)) * 1.05

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for seed in sorted(summary):
        pts = summary[seed]
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        clipped = [min(y, y_top * 0.98) for y in ys]
        ax.plot(xs, clipped, marker="o", linewidth=2.2, markersize=5, label=f"seed {seed}", color=colors.get(seed))
        for x, y in zip(xs, ys):
            if y > y_top:
                ax.plot([x], [y_top * 0.98], marker="^", color=colors.get(seed), markersize=8)
                ax.annotate(
                    f"seed {seed} outlier: {y:.2f}",
                    xy=(x, y_top * 0.98),
                    xytext=(x + 18000, y_top * 0.86),
                    fontsize=9,
                    arrowprops={"arrowstyle": "->", "lw": 1.0},
                )

    _style_axis(ax)
    ax.set_ylim(0.0, y_top)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric_label)
    ax.set_title("DreamerV3+VTA: Adjacent z distance across training (robust scale)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_windows_band(
    out_png: Path,
    windows_stats: dict[int, dict[int, tuple[float, float, float, float]]],
    metric_label: str,
    colors: dict[int, str],
):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    all_means = []
    for seed in sorted(windows_stats):
        items = sorted(windows_stats[seed].items(), key=lambda x: x[0])
        xs = [x for x, _ in items]
        p10 = [t[0] for _, t in items]
        mean = [t[1] for _, t in items]
        p50 = [t[2] for _, t in items]
        p90 = [t[3] for _, t in items]
        all_means.extend(mean)
        c = colors.get(seed)
        ax.fill_between(xs, p10, p90, alpha=0.20, color=c)
        ax.plot(xs, mean, marker="o", linewidth=2.2, markersize=5, color=c, label=f"seed {seed} mean (band=p10-p90)")
        ax.plot(xs, p50, linewidth=1.2, linestyle="--", color=c, alpha=0.65)

    # Robust y-range for readability.
    mean_arr = np.asarray(all_means, dtype=float)
    q95 = float(np.quantile(mean_arr, 0.95))
    q99 = float(np.quantile(mean_arr, 0.99))
    y_top = q95 * 1.25 if q99 > q95 * 3.0 else float(np.max(mean_arr)) * 1.05
    y_top = max(y_top, 0.05)
    ax.set_ylim(0.0, y_top)
    _style_axis(ax)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric_label)
    ax.set_title("DreamerV3+VTA: Adjacent z distance with shifted windows (line=mean, dashed=median)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--windows_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metric", default="abs_stoch_l2_mean")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
    metric_label = f"{args.metric} (z_t vs z_(t+1) L2)"

    summary = load_summary(args.summary_csv, args.metric)
    windows_stats = load_windows(args.windows_csv, args.metric)

    plot_full(args.output_dir / "z_distance_3seed_envstep400k_fullscale.png", summary, metric_label, colors)
    plot_robust(args.output_dir / "z_distance_3seed_envstep400k_robust.png", summary, metric_label, colors)
    plot_windows_band(args.output_dir / "z_distance_3seed_envstep400k_windows_p10_p90.png", windows_stats, metric_label, colors)

    print("Wrote plots:")
    print(args.output_dir / "z_distance_3seed_envstep400k_fullscale.png")
    print(args.output_dir / "z_distance_3seed_envstep400k_robust.png")
    print(args.output_dir / "z_distance_3seed_envstep400k_windows_p10_p90.png")


if __name__ == "__main__":
    main()
