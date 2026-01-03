#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import ruamel.yaml as yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gym
from torch import distributions as torchd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src_dreamerv3 import tools
from src_dreamerv3 import models


def load_config(config_names, overrides):
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    configs = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "src_dreamerv3" / "configs.yaml").read_text()
    )
    defaults = {}
    for name in ["defaults", *config_names]:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(overrides)


def pick_episode(episodes_dir, explicit_path=None):
    if explicit_path:
        return Path(explicit_path)
    candidates = list(Path(episodes_dir).glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No episodes found in {episodes_dir}")
    # Prefer the longest episode (length encoded at filename suffix)
    def ep_len(path):
        try:
            return int(path.stem.split("-")[-1])
        except ValueError:
            return 0
    return max(candidates, key=ep_len)


def compute_vta_stats(wm, data):
    wm.eval()
    with torch.no_grad():
        proc = wm.preprocess(data)
        embed = wm.encoder(proc)
        reward = proc.get("reward", None)
        post, prior = wm.dynamics.observe(embed, proc["action"], proc["is_first"], reward=reward)
        
        # For post_boundary, also need to pass reward if embed_reward mode
        if hasattr(wm.dynamics, '_posterior_input') and wm.dynamics._posterior_input == 'embed_reward' and reward is not None:
            # Concatenate reward to embed for post_boundary
            reward_inp = reward.unsqueeze(-1)  # (batch, time, 1)
            post_inp = torch.cat([embed, reward_inp], dim=-1)
            post_logits = wm.dynamics.post_boundary(post_inp)
        else:
            post_logits = wm.dynamics.post_boundary(embed)
        post_probs = torch.softmax(post_logits, dim=-1)[..., 0]

        post_abs = torchd.normal.Normal(post["abs_mean"], post["abs_std"])
        prior_abs = torchd.normal.Normal(prior["abs_mean"], prior["abs_std"])
        post_obs = torchd.normal.Normal(post["obs_mean"], post["obs_std"])
        prior_obs = torchd.normal.Normal(prior["obs_mean"], prior["obs_std"])

        abs_kl = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_abs, 1),
            torchd.independent.Independent(prior_abs, 1),
        )
        obs_kl = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_obs, 1),
            torchd.independent.Independent(prior_obs, 1),
        )

    stats = {
        "boundary": post["boundary"].squeeze(-1).detach().cpu().numpy(),
        "read_prob": post_probs.detach().cpu().numpy(),
        "seg_len": post["seg_len"].squeeze(-1).detach().cpu().numpy(),
        "seg_num": post["seg_num"].squeeze(-1).detach().cpu().numpy(),
        "abs_kl": abs_kl.detach().cpu().numpy(),
        "obs_kl": obs_kl.detach().cpu().numpy(),
    }
    return stats


def choose_window(boundary, reward, frame_delta, length, start=None, mode="boundary"):
    if start is not None:
        return int(start)
    if boundary.shape[1] <= length:
        return 0
    if mode == "reward":
        if reward is None or np.allclose(reward, 0):
            mode = "boundary"
        else:
            scores = np.convolve(np.abs(reward), np.ones(length), mode="valid")
            return int(np.argmax(scores))
    if mode == "delta":
        if frame_delta is None:
            mode = "boundary"
        else:
            scores = np.convolve(frame_delta, np.ones(length), mode="valid")
            return int(np.argmax(scores))
    if mode == "reward_or_delta":
        if reward is not None and not np.allclose(reward, 0):
            scores = np.convolve(np.abs(reward), np.ones(length), mode="valid")
            return int(np.argmax(scores))
        if frame_delta is not None:
            scores = np.convolve(frame_delta, np.ones(length), mode="valid")
            return int(np.argmax(scores))
    # Pick window with most boundaries for clearer visualization.
    counts = np.convolve(boundary[0], np.ones(length), mode="valid")
    return int(np.argmax(counts))


def render_plot(images, boundary, out_path, title):
    t_len = len(images)
    fig = plt.figure(figsize=(max(8, t_len * 0.8), 5))
    gs = fig.add_gridspec(2, t_len, height_ratios=[3, 1.2], hspace=0.35)

    # Frames with colored borders indicating boundary.
    for t in range(t_len):
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(images[t])
        ax.set_title(f"t={t}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        color = "red" if boundary[t] > 0.5 else "blue"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)

    ax = fig.add_subplot(gs[1, :])
    t = np.arange(t_len)
    colors = ["#e41a1c" if b > 0.5 else "#377eb8" for b in boundary]
    ax.bar(t, boundary, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("m (boundary)")
    ax.set_xlabel("Time Step")
    ax.set_title(title)
    # Highlight READ steps with vertical bands for clarity.
    for idx, b in enumerate(boundary):
        if b > 0.5:
            ax.axvspan(idx - 0.45, idx + 0.45, color="#e41a1c", alpha=0.1)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#e41a1c", label="READ (boundary)"),
        plt.Rectangle((0, 0), 1, 1, color="#377eb8", label="COPY (continue)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_internal_plot(stats, reward, frame_delta, start, end, out_path, title):
    boundary = stats["boundary"][0, start:end]
    read_prob = stats["read_prob"][0, start:end]
    seg_len = stats["seg_len"][0, start:end]
    seg_num = stats["seg_num"][0, start:end]
    abs_kl = stats["abs_kl"][0, start:end]
    obs_kl = stats["obs_kl"][0, start:end]
    reward_win = reward[start:end] if reward is not None else None
    delta_win = frame_delta[start:end] if frame_delta is not None else None

    t = np.arange(len(boundary))
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, read_prob, color="black", linewidth=2.0, label="READ prob")
    axes[0].scatter(
        t,
        boundary,
        color=["#e41a1c" if b > 0.5 else "#377eb8" for b in boundary],
        s=30,
        zorder=3,
    )
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("READ prob / sample")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, seg_len, color="#1f77b4", label="seg_len")
    axes[1].plot(t, seg_num, color="#ff7f0e", label="seg_num")
    axes[1].set_ylabel("Segment")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, abs_kl, color="#2ca02c", label="abs_kl")
    axes[2].plot(t, obs_kl, color="#d62728", label="obs_kl")
    axes[2].set_ylabel("KL")
    axes[2].legend(loc="upper right")

    if reward_win is not None:
        axes[3].bar(t, reward_win, color="#9467bd", alpha=0.6, label="reward")
    if delta_win is not None:
        axes[3].plot(t, delta_win, color="#8c564b", linewidth=1.5, label="frame_delta")
    axes[3].set_ylabel("Reward / Delta")
    axes[3].set_xlabel("Time Step")
    axes[3].legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_fired_plot(images, boundary, indices, out_path, title):
    if len(indices) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No READ boundaries in this episode.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    cols = min(10, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig = plt.figure(figsize=(max(8, cols * 1.2), max(2.5, rows * 1.6)))
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.2)

    for i, idx in enumerate(indices):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(images[idx])
        ax.set_title(f"t={idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#e41a1c")
            spine.set_linewidth(2.0)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--configs", nargs="+", default=["atari100k"])
    parser.add_argument("--task", default="atari_breakout")
    parser.add_argument("--episode", default=None, help="Path to a .npz episode file")
    parser.add_argument("--episodes_dir", default="train_eps")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument(
        "--window",
        default="boundary",
        choices=["boundary", "reward", "delta", "reward_or_delta"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--internal_out", default=None)
    parser.add_argument("--fired_out", default=None)
    args, overrides = parser.parse_known_args()

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = load_config(
        args.configs,
        ["--task", args.task, "--dynamics_type", "vta", "--device", device, *overrides],
    )

    logdir = Path(args.logdir)
    ckpt_path = logdir / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ep_path = pick_episode(logdir / args.episodes_dir, args.episode)
    with np.load(ep_path) as ep:
        images = ep["image"]
        actions = ep["action"]
        reward = ep["reward"] if "reward" in ep else None
        is_first = ep["is_first"]
        is_terminal = ep["is_terminal"]
        discount = ep["discount"] if "discount" in ep else None

    action_dim = actions.shape[-1]
    config.num_actions = action_dim

    obs_space = gym.spaces.Dict(
        {"image": gym.spaces.Box(0, 255, shape=images.shape[1:], dtype=np.uint8)}
    )
    act_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
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
    if reward is not None:
        data["reward"] = reward[None]
    if discount is not None:
        data["discount"] = discount[None]

    stats = compute_vta_stats(wm, data)

    frame_delta = None
    if images.shape[0] > 1:
        diff = np.abs(images[1:].astype(np.float32) - images[:-1].astype(np.float32))
        frame_delta = diff.mean(axis=(1, 2, 3))
        frame_delta = np.concatenate([[0.0], frame_delta], axis=0)

    start = choose_window(stats["boundary"], reward, frame_delta, args.length, args.start, args.window)
    end = start + args.length

    images_win = images[start:end]
    boundary_win = stats["boundary"][0, start:end]

    out_path = Path(args.out) if args.out else logdir / "boundary_grid.png"
    title = f"VTA Boundary Detection - {args.task} ({ep_path.name})"
    render_plot(images_win, boundary_win, out_path, title)
    print(out_path)

    internal_out = Path(args.internal_out) if args.internal_out else logdir / "boundary_internal.png"
    render_internal_plot(stats, reward, frame_delta, start, end, internal_out, title)
    print(internal_out)

    fired_out = Path(args.fired_out) if args.fired_out else logdir / "boundary_fired.png"
    boundary_all = stats["boundary"][0]
    fired_indices = np.where(boundary_all > 0.5)[0].tolist()
    render_fired_plot(images, boundary_all, fired_indices, fired_out, title)
    print(fired_out)


if __name__ == "__main__":
    main()
