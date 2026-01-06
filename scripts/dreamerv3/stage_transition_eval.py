#!/usr/bin/env python3
"""
Stage Transition Evaluation Script

Evaluates how well the VTA model detects stage transitions (large frame deltas).
Reports:
- Total number of stage transitions (large frame delta events) in each episode
- How many of those transitions the model detected as boundaries
- Detection rate (recall) for stage transitions
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import gym

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.dreamerv3.vta_boundary_viz import load_config, compute_vta_stats
from src_dreamerv3 import models


def visualize_stage_eval(images, frame_delta, transition_mask, boundary_mask, 
                         metrics, output_path, title=""):
    """
    Create visualization showing stage transitions vs model boundaries.
    
    Args:
        images: numpy array of shape (T, H, W, C)
        frame_delta: frame delta values
        transition_mask: boolean array for stage transitions
        boundary_mask: boolean array for model boundaries
        metrics: dict with evaluation metrics
        output_path: path to save the image
        title: title for the figure
    """
    import matplotlib.pyplot as plt
    
    T = len(images)
    transition_indices = np.where(transition_mask)[0]
    boundary_indices = np.where(boundary_mask)[0]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    # Subplot 1: Frame delta with markers
    ax1 = axes[0]
    ax1.plot(frame_delta, color='gray', alpha=0.7, linewidth=0.8, label='Frame Delta')
    ax1.fill_between(range(T), frame_delta, alpha=0.3, color='gray')
    
    # Mark stage transitions (ground truth)
    for idx in transition_indices:
        ax1.axvline(x=idx, color='green', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Mark model boundaries
    for idx in boundary_indices:
        ax1.axvline(x=idx, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    
    ax1.set_ylabel('Frame Delta')
    ax1.set_xlim(0, T)
    ax1.legend([
        plt.Line2D([0], [0], color='green', linewidth=2, label='Stage Transition (GT)'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Model Boundary'),
    ], ['Stage Transition (GT)', 'Model Boundary'], loc='upper right')
    ax1.set_title(f"{title}\nRecall: {metrics['recall']*100:.1f}%, Precision: {metrics['precision']*100:.1f}%")
    
    # Subplot 2: Timeline visualization
    ax2 = axes[1]
    
    # Draw timeline bars
    for idx in transition_indices:
        ax2.barh(1, 1, left=idx-0.5, height=0.4, color='green', alpha=0.8)
    for idx in boundary_indices:
        ax2.barh(0, 1, left=idx-0.5, height=0.4, color='red', alpha=0.8)
    
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Model Boundaries', 'Stage Transitions'])
    ax2.set_xlim(0, T)
    ax2.set_xlabel('Time Step')
    ax2.set_ylim(-0.5, 1.5)
    
    # Add stats text
    stats_text = f"Transitions: {metrics['n_transitions']} | Boundaries: {metrics['n_boundaries']} | Detected: {metrics['n_detected']}"
    ax2.text(0.02, -0.3, stats_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def detect_stage_transitions(images, method="peak", threshold=90.0, min_delta=None, peak_prominence=3.0):
    """
    Detect stage transitions based on frame delta.
    
    Args:
        images: numpy array of shape (T, H, W, C)
        method: "percentile", "absolute", or "peak"
        threshold: percentile value (if method="percentile") or absolute value
        min_delta: minimum delta value to consider (optional, for filtering noise)
        peak_prominence: for peak method, how many standard deviations above mean to be a peak
    
    Returns:
        transition_mask: boolean array of shape (T,) where True indicates a stage transition
        frame_delta: float array of shape (T,) with delta values
    """
    if images.shape[0] < 2:
        return np.zeros(images.shape[0], dtype=bool), np.zeros(images.shape[0], dtype=np.float32)
    
    # Compute frame-to-frame difference
    diff = np.abs(images[1:].astype(np.float32) - images[:-1].astype(np.float32))
    delta = diff.mean(axis=(1, 2, 3))
    frame_delta = np.concatenate([[0.0], delta], axis=0)
    
    if method == "peak":
        # Peak detection: find frames with delta significantly above the mean
        mean_delta = frame_delta.mean()
        std_delta = frame_delta.std()
        thresh_value = mean_delta + peak_prominence * std_delta
        if min_delta is not None:
            thresh_value = max(thresh_value, min_delta)
        transition_mask = frame_delta >= thresh_value
    elif method == "percentile":
        thresh_value = np.percentile(frame_delta, threshold)
        if min_delta is not None:
            thresh_value = max(thresh_value, min_delta)
        transition_mask = frame_delta >= thresh_value
    else:  # absolute
        thresh_value = threshold
        if min_delta is not None:
            thresh_value = max(thresh_value, min_delta)
        transition_mask = frame_delta >= thresh_value
    
    return transition_mask, frame_delta


def evaluate_detection(transition_mask, boundary_mask, window=1):
    """
    Evaluate how well boundaries capture stage transitions.
    
    Args:
        transition_mask: boolean array indicating ground truth transitions
        boundary_mask: boolean array indicating model-detected boundaries
        window: tolerance window for matching (boundaries within ±window frames count as a match)
    
    Returns:
        dict with evaluation metrics
    """
    n_transitions = int(transition_mask.sum())
    n_boundaries = int(boundary_mask.sum())
    
    if n_transitions == 0:
        return {
            "n_transitions": 0,
            "n_boundaries": n_boundaries,
            "n_detected": 0,
            "recall": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
        }
    
    # For each transition, check if there's a boundary within the window
    transition_indices = np.where(transition_mask)[0]
    boundary_indices = np.where(boundary_mask)[0]
    
    detected_transitions = 0
    matched_boundaries = set()
    
    for t_idx in transition_indices:
        # Check if any boundary is within window
        for b_idx in boundary_indices:
            if abs(t_idx - b_idx) <= window:
                detected_transitions += 1
                matched_boundaries.add(b_idx)
                break
    
    recall = detected_transitions / n_transitions if n_transitions > 0 else float("nan")
    precision = len(matched_boundaries) / n_boundaries if n_boundaries > 0 else float("nan")
    
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = float("nan")
    
    return {
        "n_transitions": n_transitions,
        "n_boundaries": n_boundaries,
        "n_detected": detected_transitions,
        "n_matched_boundaries": len(matched_boundaries),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VTA boundary detection on stage transitions")
    parser.add_argument("--logdir", required=True, help="Path to log directory with trained model")
    parser.add_argument("--configs", nargs="+", default=["atari100k"], help="Config names")
    parser.add_argument("--task", default="atari_private_eye", help="Task name")
    parser.add_argument("--episodes_dir", default="train_eps", help="Directory containing episodes")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--episode_file", nargs="+", default=None, help="Specific episode file(s) to evaluate (takes precedence over --episodes)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode selection")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile threshold for stage transitions (used with --method percentile)")
    parser.add_argument("--method", default="peak", choices=["peak", "percentile", "absolute"], help="Method for detecting stage transitions")
    parser.add_argument("--prominence", type=float, default=3.0, help="Standard deviations above mean for peak detection")
    parser.add_argument("--window", type=int, default=2, help="Tolerance window for matching (frames)")
    parser.add_argument("--max_seg_len", type=int, default=None, help="Override max_seg_len for evaluation (default: use trained value)")
    parser.add_argument("--force_scale", type=float, default=None, help="Override boundary_force_scale (default: use trained value, 0 disables forced boundaries)")
    parser.add_argument("--boundary_temp", type=float, default=None, help="Override boundary temperature (higher = more selective)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-episode details")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization images")
    parser.add_argument("--output_dir", default=None, help="Output directory for visualizations (default: logdir/stage_eval)")
    args, overrides = parser.parse_known_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = load_config(
        args.configs,
        ["--task", args.task, "--dynamics_type", "vta", "--device", device, *overrides],
    )

    logdir = Path(args.logdir)
    ckpt_path = logdir / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # Load episodes
    eps_dir = logdir / args.episodes_dir
    
    # If specific episode files are provided, use those
    if args.episode_file:
        eps_files = []
        for ep_name in args.episode_file:
            # Support both filename only or full path
            if Path(ep_name).exists():
                eps_files.append(Path(ep_name))
            else:
                ep_path = eps_dir / ep_name
                if not ep_path.suffix:
                    ep_path = ep_path.with_suffix('.npz')
                if ep_path.exists():
                    eps_files.append(ep_path)
                else:
                    print(f"Warning: Episode file not found: {ep_name}")
        if not eps_files:
            raise FileNotFoundError(f"No valid episode files found from: {args.episode_file}")
    else:
        eps_files = sorted(eps_dir.glob("*.npz"), key=lambda p: -int(p.stem.split("-")[-1]) if p.stem.split("-")[-1].isdigit() else 0)
        
        if not eps_files:
            raise FileNotFoundError(f"No episodes found in {eps_dir}")
        
        # Select episodes
        rng = np.random.default_rng(args.seed)
        if args.episodes < len(eps_files):
            indices = rng.choice(len(eps_files), size=args.episodes, replace=False)
            eps_files = [eps_files[i] for i in sorted(indices)]
        else:
            eps_files = eps_files[:args.episodes]

    print(f"\n{'='*60}")
    print(f"Stage Transition Detection Evaluation")
    print(f"{'='*60}")
    print(f"Log directory: {logdir}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Episodes: {len(eps_files)}")
    if args.method == "peak":
        print(f"Detection method: peak (prominence={args.prominence} std)")
    elif args.method == "percentile":
        print(f"Detection method: percentile ({args.percentile}th)")
    else:
        print(f"Detection method: absolute (threshold={args.percentile})")
    print(f"Matching window: ±{args.window} frames")
    print(f"{'='*60}\n")

    all_results = []

    for ep_path in eps_files:
        with np.load(ep_path) as ep:
            images = ep["image"]
            actions = ep["action"]
            is_first = ep["is_first"]
            is_terminal = ep["is_terminal"]
            discount = ep.get("discount", None)

        # Detect stage transitions in this episode
        transition_mask, frame_delta = detect_stage_transitions(
            images, method=args.method, threshold=args.percentile, peak_prominence=args.prominence
        )

        # Load model and get boundary predictions
        config.num_actions = actions.shape[-1]
        obs_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, shape=images.shape[1:], dtype=np.uint8)}
        )
        act_space = gym.spaces.Box(low=0, high=1, shape=(actions.shape[-1],), dtype=np.float32)
        act_space.discrete = True

        wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(device)
        state = torch.load(ckpt_path, map_location=device)["agent_state_dict"]
        wm_state = {k[len("_wm."):]: v for k, v in state.items() if k.startswith("_wm.")}
        wm_state = {k.replace("_orig_mod.", ""): v for k, v in wm_state.items()}
        wm.load_state_dict(wm_state, strict=True)

        # Override max_seg_len if specified
        if args.max_seg_len is not None and hasattr(wm, 'dynamics') and hasattr(wm.dynamics, '_max_seg_len'):
            original_msl = wm.dynamics._max_seg_len
            wm.dynamics._max_seg_len = args.max_seg_len
            if args.verbose and ep_path == eps_files[0]:
                print(f"Overriding max_seg_len: {original_msl} -> {args.max_seg_len}")

        # Override force_scale if specified
        if args.force_scale is not None and hasattr(wm, 'dynamics') and hasattr(wm.dynamics, '_boundary_force_scale'):
            original_fs = wm.dynamics._boundary_force_scale
            wm.dynamics._boundary_force_scale = args.force_scale
            if args.verbose and ep_path == eps_files[0]:
                print(f"Overriding force_scale: {original_fs} -> {args.force_scale}")

        # Override boundary_temp if specified
        if args.boundary_temp is not None and hasattr(wm, 'dynamics') and hasattr(wm.dynamics, '_boundary_temp'):
            original_temp = wm.dynamics._boundary_temp
            wm.dynamics._boundary_temp = args.boundary_temp
            if args.verbose and ep_path == eps_files[0]:
                print(f"Overriding boundary_temp: {original_temp} -> {args.boundary_temp}\n")

        data = {
            "image": images[None],
            "action": actions[None],
            "is_first": is_first[None],
            "is_terminal": is_terminal[None],
        }
        if discount is not None:
            data["discount"] = discount[None]

        stats = compute_vta_stats(wm, data)
        boundary_mask = stats["boundary"][0] > 0.5

        # Evaluate
        result = evaluate_detection(transition_mask, boundary_mask, window=args.window)
        result["episode"] = ep_path.name
        result["length"] = len(images)
        all_results.append(result)

        if args.verbose:
            print(f"Episode: {ep_path.name} (length={len(images)})")
            print(f"  Stage transitions: {result['n_transitions']}")
            print(f"  Model boundaries:  {result['n_boundaries']}")
            print(f"  Detected:          {result['n_detected']} / {result['n_transitions']}")
            print(f"  Recall:            {result['recall']:.2%}" if not np.isnan(result['recall']) else "  Recall:            N/A")
            print(f"  Precision:         {result['precision']:.2%}" if not np.isnan(result['precision']) else "  Precision:         N/A")
            print()

        # Generate visualization if requested
        if args.visualize:
            output_dir = Path(args.output_dir) if args.output_dir else logdir / "stage_eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            viz_path = output_dir / f"{ep_path.stem}_stage_eval.png"
            visualize_stage_eval(
                images, frame_delta, transition_mask, boundary_mask,
                result, viz_path,
                title=f"Stage Transition Evaluation - {ep_path.stem}"
            )

    # Aggregate results
    total_transitions = sum(r["n_transitions"] for r in all_results)
    total_boundaries = sum(r["n_boundaries"] for r in all_results)
    total_detected = sum(r["n_detected"] for r in all_results)
    
    avg_recall = total_detected / total_transitions if total_transitions > 0 else float("nan")
    
    matched = sum(r["n_matched_boundaries"] for r in all_results)
    avg_precision = matched / total_boundaries if total_boundaries > 0 else float("nan")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes evaluated:     {len(all_results)}")
    print(f"Total stage transitions:      {total_transitions}")
    print(f"Total model boundaries:       {total_boundaries}")
    print(f"Total transitions detected:   {total_detected}")
    print(f"{'='*60}")
    print(f"Overall Recall:    {avg_recall:.2%}" if not np.isnan(avg_recall) else "Overall Recall:    N/A")
    print(f"  (= {total_detected} / {total_transitions} transitions captured by model)")
    print(f"Overall Precision: {avg_precision:.2%}" if not np.isnan(avg_precision) else "Overall Precision: N/A")
    print(f"  (= {matched} / {total_boundaries} model boundaries match a transition)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
