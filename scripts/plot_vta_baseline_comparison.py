#!/usr/bin/env python3
"""Plot VTA baseline comparison: z_t Read distance and s_t all-step distance.

Usage:
    python scripts/plot_vta_baseline_comparison.py \
        --summary_csv /tmp/vta_prior_comparison/vta_z_summary.csv \
        --out_dir /tmp/vta_prior_comparison

    # Custom label mapping (logdir_basename=Label):
    python scripts/plot_vta_baseline_comparison.py \
        --summary_csv /tmp/vta_prior_comparison/vta_z_summary.csv \
        --labels fixed_k20="Fixed (k=20)" atari_krull_seed2="Learned VTA"
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2",
    "#CCB974", "#64B5CD", "#E377C2", "#7F7F7F",
]
DEFAULT_MARKERS = ["s", "^", "D", "o", "v", "P", "X", "*"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary_csv", type=str, required=True, help="Path to vta_z_summary.csv")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same as CSV)")
    p.add_argument("--labels", nargs="*", default=[], help="Label overrides: logdir_basename=Label")
    p.add_argument("--action_repeat", type=int, default=4, help="Action repeat for env step conversion")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--figsize", type=float, nargs=2, default=[8, 5.5])
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse label overrides
    label_map = {}
    for item in args.labels:
        k, v = item.split("=", 1)
        label_map[k] = v

    # Read CSV
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Organize by model
    models = {}
    model_order = []
    for r in rows:
        name = r["logdir"].split("/")[-1]
        m = re.search(r"step-(\d+)", r["ckpt"])
        env_step = int(m.group(1)) * args.action_repeat if m else -1
        if name not in models:
            models[name] = {"env_steps": [], "obs_stoch_l2_mean": [], "abs_stoch_l2_read_mean": []}
            model_order.append(name)
        models[name]["env_steps"].append(env_step)
        models[name]["obs_stoch_l2_mean"].append(float(r["obs_stoch_l2_mean"]))
        models[name]["abs_stoch_l2_read_mean"].append(float(r["abs_stoch_l2_read_mean"]))

    # Assign labels, colors, markers
    labels = {name: label_map.get(name, name) for name in model_order}
    colors = {name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, name in enumerate(model_order)}
    markers = {name: DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)] for i, name in enumerate(model_order)}

    # Format x-tick labels
    all_steps = sorted(set(s for d in models.values() for s in d["env_steps"]))
    xtick_labels = [f"{s // 1000}k" for s in all_steps]

    def make_plot(metric_key, ylabel, title, out_name):
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        for name in model_order:
            d = models[name]
            ax.plot(
                d["env_steps"], d[metric_key],
                label=labels[name], color=colors[name], marker=markers[name],
                linewidth=2.5, markersize=9,
            )
        ax.set_xlabel("Environment Steps", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=11)
        ax.set_xticks(all_steps)
        ax.set_xticklabels(xtick_labels)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / out_name
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    make_plot(
        "abs_stoch_l2_read_mean",
        r"Read-Conditioned $\|z_{t+1} - z_t\|_2$",
        r"Read-Conditioned $z_t$ Distance (Prior)",
        "zt_read_distance.png",
    )
    make_plot(
        "obs_stoch_l2_mean",
        r"All-Step $\|s_{t+1} - s_t\|_2$",
        r"All-Step $s_t$ Distance (Prior)",
        "st_allstep_distance.png",
    )


if __name__ == "__main__":
    main()
