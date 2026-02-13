#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import seaborn as sns

def extract_step(ckpt_path):
    # Extract numeric step from checkpoint filename (e.g., step-000050000.pt)
    m = re.search(r"step-(\d+)\.pt", str(ckpt_path))
    if m:
        return int(m.group(1))
    if "latest.pt" in str(ckpt_path):
        # Determine step from latest? Ideally avoiding this unless we know the step.
        # But usually latest.pt doesn't have the step in name.
        # For this plot, let's skip latest.pt if it doesn't have step info, 
        # or assume it's the final step if we can't find others.
        # Actually vta_ckpt_sweep_eval might not provide step for latest.
        return -1 
    return 0

def infer_condition(logdir):
    if "fixed" in logdir:
        return "Fixed (k=20)"
    if "bernoulli" in logdir:
        return "Bernoulli (p=0.05)"
    if "exponential" in logdir:
        return "Exponential (lambda=0.05)"
    return "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to vta_z_summary.csv")
    parser.add_argument("--output", default="vta_baseline_comparison.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Extract steps
    df["step"] = df["ckpt"].apply(extract_step)
    # Filter out invalid steps (-1 for latest.pt if we want to ignore it to avoid confusion, 
    # or if we want to include it we need to know the step. 
    # vta_ckpt_sweep_eval doesn't output step for latest.pt directly in ckpt name. 
    # Let's filter > 0)
    df = df[df["step"] >= 0]

    # Infer condition
    df["Condition"] = df["logdir"].apply(infer_condition)

    # Sort
    df = df.sort_values("step")

    # Plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Read-Conditioned Latent Transition Distance (abs_l2_read_mean)
    sns.lineplot(data=df, x="step", y="abs_l2_read_mean", hue="Condition", marker="o", ax=axes[0])
    axes[0].set_title("Read-Conditioned Latent Transition Distance (z)")
    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("L2 Distance")

    # 2. Mean Obs-level Transition Distance (obs_l2_mean)
    sns.lineplot(data=df, x="step", y="obs_l2_mean", hue="Condition", marker="o", ax=axes[1])
    axes[1].set_title("Observation-Level Transition Distance (s)")
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("L2 Distance")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved comparison plot to {args.output}")

if __name__ == "__main__":
    main()
