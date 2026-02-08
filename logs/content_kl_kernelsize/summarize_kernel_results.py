#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def _safe_float(x):
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v


def _nanmean(values):
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def read_last_metrics(logdir: Path):
    metrics_path = logdir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    last = {}
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Keep latest scalar values only.
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    last[key] = float(value)
    return last


def read_single_row_csv(path: Path):
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    return rows[-1]


def read_series_mean(path: Path, key: str):
    if not path.exists():
        return math.nan
    vals = []
    for row in csv.DictReader(path.open()):
        vals.append(_safe_float(row.get(key, "nan")))
    return _nanmean(vals)


def infer_kernel(logdir: Path):
    name = logdir.name
    if "_k" in name:
        try:
            return int(name.split("_k", 1)[1].split("_", 1)[0])
        except Exception:
            pass
    cfg = logdir / "config.yaml"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.startswith("vta_post_boundary_kernel_size:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except Exception:
                    return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logdirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    fieldnames = [
        "run_dir",
        "kernel_size",
        "ctx_kl_mean",
        "abs_kl_post_prior_mean",
        "boundary_rate",
        "read_prob_mean",
        "abs_l2_mean",
        "abs_l2_read_mean",
        "obs_l2_mean",
        "obs_l2_read_mean",
        "metrics_kl_mask_last",
        "metrics_model_loss_last",
        "metrics_reward_last",
    ]

    rows = []
    for logdir in logdirs:
        analysis = logdir / "analysis"
        boundary_stats = read_single_row_csv(analysis / "boundary_stats.csv")
        metrics = read_last_metrics(logdir)

        row = {
            "run_dir": str(logdir),
            "kernel_size": infer_kernel(logdir),
            "ctx_kl_mean": read_series_mean(analysis / "vta_abs_ctx_kl_series.csv", "ctx_kl"),
            "abs_kl_post_prior_mean": read_series_mean(
                analysis / "vta_abs_ctx_kl_series.csv", "abs_kl_post_prior"
            ),
            "boundary_rate": _safe_float(boundary_stats.get("boundary_rate", "nan")),
            "read_prob_mean": _safe_float(boundary_stats.get("read_prob_mean", "nan")),
            "abs_l2_mean": _safe_float(boundary_stats.get("abs_l2_mean", "nan")),
            "abs_l2_read_mean": _safe_float(boundary_stats.get("abs_l2_read_mean", "nan")),
            "obs_l2_mean": _safe_float(boundary_stats.get("obs_l2_mean", "nan")),
            "obs_l2_read_mean": _safe_float(boundary_stats.get("obs_l2_read_mean", "nan")),
            "metrics_kl_mask_last": _safe_float(metrics.get("kl_mask", math.nan)),
            "metrics_model_loss_last": _safe_float(metrics.get("model_loss", math.nan)),
            "metrics_reward_last": _safe_float(metrics.get("reward", math.nan)),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["kernel_size"] is None, r["kernel_size"]))

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
