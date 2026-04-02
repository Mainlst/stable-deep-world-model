#!/usr/bin/env python3
"""
Boundary Frames Visualization

Visualizes the frames around each detected boundary:
- Frame before boundary (t-1)
- Boundary frame (t)
- Frame after boundary (t+1)

Outputs a grid image showing all boundaries in an episode.
"""
import argparse
import importlib
import inspect
from pathlib import Path
import sys

import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ruamel.yaml as yaml
from torch import distributions as torchd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def _backend_paths(backend):
    if backend == "director_vta":
        return (
            "src_director_vta.tools",
            "src_director_vta.models",
            ROOT / "src_director_vta" / "configs.yaml",
        )
    return (
        "src_dreamerv3.tools",
        "src_dreamerv3.models",
        ROOT / "src_dreamerv3" / "configs.yaml",
    )


def _detect_backend(agent_state_dict):
    # Director-VTA checkpoints include GoalAE modules under world model.
    if any(("goal_enc" in k or "goal_dec" in k) for k in agent_state_dict.keys()):
        return "director_vta"
    return "dreamerv3"


def _load_named_config(config_names, overrides, tools_module, configs_path):
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    configs = yaml.safe_load(configs_path.read_text())
    defaults = {}
    for name in ["defaults", *config_names]:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools_module.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(overrides)


def _load_logdir_config(logdir, overrides, tools_module):
    config_path = logdir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file in logdir: {config_path}")
    yaml_loader = yaml.YAML(typ="unsafe", pure=True)
    defaults = yaml_loader.load(config_path.read_text())
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools_module.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(overrides)


def _stack_time(states):
    keys = states[0].keys()
    stacked = {}
    for key in keys:
        stacked[key] = torch.stack([s[key] for s in states], dim=1)
    return stacked


def _prior_rollout(wm, proc, embed):
    action = proc["action"]
    is_first = proc["is_first"]
    batch_size, seq_len = embed.shape[:2]
    state = None
    posts, priors = [], []
    obs_step_sig = inspect.signature(wm.dynamics.obs_step)
    has_apply_constraints = "apply_boundary_constraints" in obs_step_sig.parameters

    for t in range(seq_len):
        kwargs = dict(post_boundary_logit=None, sample=True)
        if has_apply_constraints:
            # Match online evaluation path in director_vta (training=False).
            kwargs["apply_boundary_constraints"] = False
        post, prior = wm.dynamics.obs_step(
            state,
            action[:, t],
            embed[:, t],
            is_first[:, t],
            **kwargs,
        )
        posts.append(post)
        priors.append(prior)
        state = post
    return _stack_time(posts), _stack_time(priors)


def compute_vta_stats(wm, data, tools_module, boundary_source="posterior"):
    wm.eval()
    with torch.no_grad():
        proc = wm.preprocess(data)
        embed = wm.encoder(proc)
        reward = proc.get("reward", None)
        if boundary_source == "prior":
            post, prior = _prior_rollout(wm, proc, embed)
        else:
            post, prior = wm.dynamics.observe(embed, proc["action"], proc["is_first"], reward=reward)

        if boundary_source == "prior":
            boundary = prior["boundary"]
            read_prob = torch.softmax(prior["boundary_logit"], dim=-1)[..., 0]
        else:
            # For post_boundary, also need to pass reward if embed_reward mode.
            if hasattr(wm.dynamics, "_posterior_input") and wm.dynamics._posterior_input == "embed_reward" and reward is not None:
                reward_inp = reward.unsqueeze(-1)  # (batch, time, 1)
                post_inp = torch.cat([embed, reward_inp], dim=-1)
                post_logits = wm.dynamics.post_boundary(post_inp)
            else:
                post_logits = wm.dynamics.post_boundary(embed)
            boundary = post["boundary"]
            read_prob = torch.softmax(post_logits, dim=-1)[..., 0]

        post_abs = torchd.normal.Normal(post["abs_mean"], post["abs_std"])
        prior_abs = torchd.normal.Normal(prior["abs_mean"], prior["abs_std"])
        abs_kl = torchd.kl.kl_divergence(
            torchd.independent.Independent(post_abs, 1),
            torchd.independent.Independent(prior_abs, 1),
        )

        if "obs_logit" in post and "obs_logit" in prior:
            unimix = getattr(wm.dynamics, "_unimix_ratio", 0.0)
            post_obs = torchd.independent.Independent(
                tools_module.OneHotDist(post["obs_logit"], unimix_ratio=unimix), 1
            )
            prior_obs = torchd.independent.Independent(
                tools_module.OneHotDist(prior["obs_logit"], unimix_ratio=unimix), 1
            )
        else:
            post_obs = torchd.normal.Normal(post["obs_mean"], post["obs_std"])
            prior_obs = torchd.normal.Normal(prior["obs_mean"], prior["obs_std"])
            post_obs = torchd.independent.Independent(post_obs, 1)
            prior_obs = torchd.independent.Independent(prior_obs, 1)
        obs_kl = torchd.kl.kl_divergence(post_obs, prior_obs)

    stats = {
        "boundary": boundary.squeeze(-1).detach().cpu().numpy(),
        "read_prob": read_prob.detach().cpu().numpy(),
        "seg_len": post["seg_len"].squeeze(-1).detach().cpu().numpy(),
        "seg_num": post["seg_num"].squeeze(-1).detach().cpu().numpy(),
        "abs_kl": abs_kl.detach().cpu().numpy(),
        "obs_kl": obs_kl.detach().cpu().numpy(),
    }
    return stats


def create_boundary_visualization(images, boundary_indices, frame_delta, output_path, 
                                  context=1, max_boundaries=20, title=""):
    """
    Create a visualization showing frames around each boundary.
    
    Args:
        images: numpy array of shape (T, H, W, C)
        boundary_indices: indices where boundaries occur
        frame_delta: frame delta values
        output_path: path to save the image
        context: number of frames before/after to show
        max_boundaries: maximum number of boundaries to show
        title: title for the figure
    """
    n_boundaries = min(len(boundary_indices), max_boundaries)
    if n_boundaries == 0:
        print("No boundaries to visualize")
        return
    
    n_frames = 2 * context + 1  # e.g., context=1 -> 3 frames (t-1, t, t+1)
    
    fig, axes = plt.subplots(n_boundaries, n_frames, figsize=(n_frames * 2, n_boundaries * 2))
    
    # Handle single boundary case
    if n_boundaries == 1:
        axes = axes.reshape(1, -1)
    
    for i, b_idx in enumerate(boundary_indices[:max_boundaries]):
        for j, offset in enumerate(range(-context, context + 1)):
            frame_idx = b_idx + offset
            ax = axes[i, j]
            
            if 0 <= frame_idx < len(images):
                ax.imshow(images[frame_idx])
                
                # Highlight the boundary frame
                if offset == 0:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                    delta_val = frame_delta[frame_idx] if frame_idx < len(frame_delta) else 0
                    ax.set_title(f"t={frame_idx}\n(Δ={delta_val:.1f})", fontsize=8, color='red', fontweight='bold')
                else:
                    delta_val = frame_delta[frame_idx] if frame_idx < len(frame_delta) else 0
                    ax.set_title(f"t={frame_idx}\n(Δ={delta_val:.1f})", fontsize=8)
            else:
                ax.set_facecolor('lightgray')
                ax.set_title(f"N/A", fontsize=8)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add column labels (positioned above frame titles)
    col_labels = [f"t-{context-j}" if j < context else ("Boundary" if j == context else f"t+{j-context}") 
                  for j in range(n_frames)]
    for j, label in enumerate(col_labels):
        axes[0, j].annotate(label, xy=(0.5, 1.35), xycoords='axes fraction',
                           ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize frames around detected boundaries")
    parser.add_argument("--logdir", required=True, help="Path to log directory with trained model")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "dreamerv3", "director_vta"],
        help="Model backend to load (auto detects from checkpoint).",
    )
    parser.add_argument(
        "--config_source",
        default="auto",
        choices=["auto", "named", "logdir"],
        help="Config source: named configs.yaml or logdir/config.yaml.",
    )
    parser.add_argument("--configs", nargs="+", default=["atari100k"], help="Config names")
    parser.add_argument("--task", default="atari_private_eye", help="Task name")
    parser.add_argument("--episodes_dir", default="train_eps", help="Directory containing episodes")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode selection")
    parser.add_argument("--context", type=int, default=1, help="Number of frames before/after boundary to show")
    parser.add_argument("--max_boundaries", type=int, default=30, help="Maximum number of boundaries to show per episode")
    parser.add_argument("--output_dir", default=None, help="Output directory for images (default: logdir/boundary_viz)")
    parser.add_argument("--max_seg_len", type=int, default=None, help="Override max_seg_len for evaluation")
    parser.add_argument("--force_scale", type=float, default=None, help="Override boundary_force_scale (0 disables forced boundaries)")
    parser.add_argument("--boundary_temp", type=float, default=None, help="Override boundary temperature (higher = more selective)")
    parser.add_argument(
        "--boundary_source",
        default="posterior",
        choices=["posterior", "prior"],
        help="Boundary source to visualize: posterior (training path) or prior (inference path).",
    )
    args, overrides = parser.parse_known_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logdir = Path(args.logdir)
    ckpt_path = logdir / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")["agent_state_dict"]

    backend = args.backend if args.backend != "auto" else _detect_backend(state)
    tools_mod_path, models_mod_path, configs_path = _backend_paths(backend)
    tools_mod = importlib.import_module(tools_mod_path)
    models_mod = importlib.import_module(models_mod_path)

    use_logdir_cfg = (
        args.config_source == "logdir" or
        (args.config_source == "auto" and (logdir / "config.yaml").exists())
    )
    if use_logdir_cfg:
        config = _load_logdir_config(logdir, overrides, tools_mod)
    else:
        config = _load_named_config(
            args.configs,
            ["--task", args.task, "--dynamics_type", "vta", "--device", device, *overrides],
            tools_mod,
            configs_path,
        )
    config.task = args.task
    config.dynamics_type = "vta"
    config.device = device

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else logdir / "boundary_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load episodes
    eps_dir = logdir / args.episodes_dir
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
    print(f"Boundary Frames Visualization")
    print(f"{'='*60}")
    print(f"Log directory: {logdir}")
    print(f"Output directory: {output_dir}")
    print(f"Backend: {backend}")
    print(f"Config source: {'logdir/config.yaml' if use_logdir_cfg else str(configs_path)}")
    print(f"Boundary source: {args.boundary_source}")
    print(f"Episodes: {len(eps_files)}")
    print(f"Context frames: ±{args.context}")
    print(f"{'='*60}\n")

    for ep_path in eps_files:
        with np.load(ep_path) as ep:
            images = ep["image"]
            actions = ep["action"]
            is_first = ep["is_first"]
            is_terminal = ep["is_terminal"]
            discount = ep.get("discount", None)

        # Compute frame delta
        if images.shape[0] < 2:
            frame_delta = np.zeros(images.shape[0], dtype=np.float32)
        else:
            diff = np.abs(images[1:].astype(np.float32) - images[:-1].astype(np.float32))
            delta = diff.mean(axis=(1, 2, 3))
            frame_delta = np.concatenate([[0.0], delta], axis=0)

        # Load model and get boundary predictions
        config.num_actions = actions.shape[-1]
        obs_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(0, 255, shape=images.shape[1:], dtype=np.uint8)}
        )
        act_space = gym.spaces.Box(low=0, high=1, shape=(actions.shape[-1],), dtype=np.float32)
        act_space.discrete = True

        wm = models_mod.WorldModel(obs_space, act_space, step=0, config=config).to(device)
        wm_state = {k[len("_wm."):]: v for k, v in state.items() if k.startswith("_wm.")}
        wm_state = {k.replace("_orig_mod.", ""): v for k, v in wm_state.items()}
        wm.load_state_dict(wm_state, strict=True)

        # Override max_seg_len if specified
        if args.max_seg_len is not None and hasattr(wm, 'dynamics') and hasattr(wm.dynamics, '_max_seg_len'):
            wm.dynamics._max_seg_len = args.max_seg_len

        # Override force_scale if specified
        if args.force_scale is not None and hasattr(wm, 'dynamics') and hasattr(wm.dynamics, '_boundary_force_scale'):
            wm.dynamics._boundary_force_scale = args.force_scale

        # Override boundary_temp if specified
        if args.boundary_temp is not None and hasattr(wm, 'dynamics') and hasattr(wm.dynamics, '_boundary_temp'):
            wm.dynamics._boundary_temp = args.boundary_temp

        data = {
            "image": images[None],
            "action": actions[None],
            "is_first": is_first[None],
            "is_terminal": is_terminal[None],
        }
        if discount is not None:
            data["discount"] = discount[None]

        stats = compute_vta_stats(wm, data, tools_mod, boundary_source=args.boundary_source)
        boundary_mask = stats["boundary"][0] > 0.5
        boundary_indices = np.where(boundary_mask)[0]

        print(f"Episode: {ep_path.name}")
        print(f"  Length: {len(images)}")
        print(f"  Boundaries detected: {len(boundary_indices)}")

        # Create visualization
        output_path = output_dir / f"{ep_path.stem}_boundaries.png"
        create_boundary_visualization(
            images, 
            boundary_indices, 
            frame_delta,
            output_path,
            context=args.context,
            max_boundaries=args.max_boundaries,
            title=(
                f"Boundary Detection ({args.boundary_source}) - {ep_path.stem}\n"
                f"{len(boundary_indices)} boundaries in {len(images)} frames"
            ),
        )


if __name__ == "__main__":
    main()
