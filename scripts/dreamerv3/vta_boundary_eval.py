#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import gym

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src_dreamerv3 import models
from vta_boundary_viz import load_config, compute_vta_stats, pick_episode


def auc_score(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    neg = labels == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Handle ties by averaging ranks.
    unique_scores, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, count in enumerate(counts):
            if count <= 1:
                continue
            mask = inv == idx
            ranks[mask] = ranks[mask].mean()
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=1) if a.size > 1 else 0.0
    var_b = b.var(ddof=1) if b.size > 1 else 0.0
    pooled = (var_a + var_b) / 2.0
    if pooled == 0:
        return float("nan")
    return float((mean_a - mean_b) / np.sqrt(pooled))


def permutation_test(event_mask, boundary_mask, repeats=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    boundary_idx = np.where(boundary_mask)[0]
    k = boundary_idx.size
    if k == 0:
        return float("nan"), float("nan")
    observed = event_mask[boundary_mask].mean()
    n = len(event_mask)
    counts = []
    for _ in range(repeats):
        rand_idx = rng.choice(n, size=k, replace=False)
        counts.append(event_mask[rand_idx].mean())
    counts = np.asarray(counts)
    p_value = (np.sum(counts >= observed) + 1) / (repeats + 1)
    return float(observed), float(p_value)


def periodic_baseline(event_mask, boundary_count):
    n = len(event_mask)
    if boundary_count <= 0 or n == 0:
        return float("nan")
    period = max(1, int(round(n / boundary_count)))
    idx = np.arange(0, n, period)[:boundary_count]
    return float(event_mask[idx].mean())


def compute_frame_delta(images):
    if images.shape[0] < 2:
        return np.zeros(images.shape[0], dtype=np.float32)
    diff = np.abs(images[1:].astype(np.float32) - images[:-1].astype(np.float32))
    delta = diff.mean(axis=(1, 2, 3))
    return np.concatenate([[0.0], delta], axis=0)


def load_episode_paths(logdir, episodes_dir, limit, seed):
    eps = list((logdir / episodes_dir).glob("*.npz"))
    if not eps:
        raise FileNotFoundError(f"No episodes found in {logdir / episodes_dir}")
    # Prefer longer episodes (length encoded at suffix)
    def ep_len(path):
        try:
            return int(path.stem.split("-")[-1])
        except ValueError:
            return 0
    eps = sorted(eps, key=ep_len, reverse=True)
    if limit and limit < len(eps):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(eps), size=limit, replace=False)
        eps = [eps[i] for i in sorted(pick)]
    return eps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--configs", nargs="+", default=["atari100k"])
    parser.add_argument("--task", default="atari_private_eye")
    parser.add_argument("--episodes_dir", default="train_eps")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta_percentile", type=float, default=90.0)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--out_json", default=None)
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

    episode_paths = load_episode_paths(logdir, args.episodes_dir, args.episodes, args.seed)
    results = []

    for ep_path in episode_paths:
        with np.load(ep_path) as ep:
            images = ep["image"]
            actions = ep["action"]
            reward = ep["reward"] if "reward" in ep else None
            is_first = ep["is_first"]
            is_terminal = ep["is_terminal"]
            discount = ep["discount"] if "discount" in ep else None

        config.num_actions = actions.shape[-1]
        obs_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, shape=images.shape[1:], dtype=np.uint8)}
        )
        act_space = gym.spaces.Box(low=0, high=1, shape=(actions.shape[-1],), dtype=np.float32)
        act_space.discrete = True

        wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(device)
        state = torch.load(ckpt_path, map_location=device)["agent_state_dict"]
        wm_state = {k[len("_wm."):]: v for k, v in state.items() if k.startswith("_wm.")}
        # Remove _orig_mod. prefix from torch.compile() saved checkpoints
        wm_state = {k.replace("_orig_mod.", ""): v for k, v in wm_state.items()}
        wm.load_state_dict(wm_state, strict=True)

        data = {
            "image": images[None],
            "action": actions[None],
            "is_first": is_first[None],
            "is_terminal": is_terminal[None],
        }
        if discount is not None:
            data["discount"] = discount[None]

        stats = compute_vta_stats(wm, data)
        boundary = stats["boundary"][0] > 0.5
        read_prob = stats["read_prob"][0]
        frame_delta = compute_frame_delta(images)

        reward_mask = np.zeros_like(boundary, dtype=bool)
        if reward is not None:
            reward_mask = np.abs(reward) > 0

        delta_threshold = np.percentile(frame_delta, args.delta_percentile)
        delta_mask = frame_delta >= delta_threshold
        reward_or_delta = reward_mask | delta_mask

        def summarize_event(name, event_mask):
            boundary_count = int(boundary.sum())
            non_boundary_count = int((~boundary).sum())
            event_rate = float(event_mask.mean())
            event_at_boundary = float(event_mask[boundary].mean()) if boundary_count else float("nan")
            event_at_non = float(event_mask[~boundary].mean()) if non_boundary_count else float("nan")
            lift = event_at_boundary / event_rate if event_rate > 0 else float("nan")
            auc = auc_score(read_prob, event_mask.astype(int))
            observed, p_value = permutation_test(
                event_mask.astype(float), boundary, repeats=args.permutations
            )
            periodic = periodic_baseline(event_mask.astype(float), boundary_count)
            return {
                "event": name,
                "event_rate": event_rate,
                "boundary_rate": float(boundary.mean()),
                "event_at_boundary": event_at_boundary,
                "event_at_nonboundary": event_at_non,
                "lift_over_event_rate": lift,
                "auc_read_prob": auc,
                "perm_observed": observed,
                "perm_p_value": p_value,
                "periodic_baseline": periodic,
            }

        delta_boundary = frame_delta[boundary]
        delta_non = frame_delta[~boundary]
        delta_summary = {
            "frame_delta_mean_boundary": float(delta_boundary.mean()) if delta_boundary.size else float("nan"),
            "frame_delta_mean_nonboundary": float(delta_non.mean()) if delta_non.size else float("nan"),
            "frame_delta_cohen_d": cohen_d(delta_boundary, delta_non),
            "frame_delta_percentile": float(args.delta_percentile),
        }

        results.append(
            {
                "episode": ep_path.name,
                "length": int(images.shape[0]),
                "boundary_count": int(boundary.sum()),
                "events": [
                    summarize_event("reward", reward_mask),
                    summarize_event("delta", delta_mask),
                    summarize_event("reward_or_delta", reward_or_delta),
                ],
                "delta_summary": delta_summary,
            }
        )

    out = {"logdir": str(logdir), "episodes": results}
    text = json.dumps(out, indent=2, ensure_ascii=True)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    main()
