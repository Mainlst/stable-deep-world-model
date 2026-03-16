
import sys
import pathlib
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml as yaml

# Add current directory to path so we can import modules
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src_dreamerv3 import dreamer
from src_dreamerv3 import tools
from src_dreamerv3 import models
from src_dreamerv3 import hierarchical_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--task', type=str, required=True)
    args, remaining = parser.parse_known_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    
    # --- Config Loading Logic (ported from dreamer.py) ---
    configs_yaml = yaml.safe_load((pathlib.Path(__file__).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs_yaml[name])
        
    # Override with task and logdir from args
    defaults['task'] = args.task
    defaults['logdir'] = str(logdir)
    
    # Parse remaining args to override defaults
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    config = parser.parse_args(remaining)
    # Force Director mode for visualization
    config.use_director = True
    config.dynamics_type = 'rssm'
    # -----------------------------------------------------

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = str(device)
    
    print(f"Loading agent from {logdir}...")
    print(f"Device: {device}")
    
    # Reconstruct agent components
    print("Reconstructing agent components...")
    
    print(f"Device: {device}")
    
    print("Initializing environment to get spaces...")
    from src_dreamerv3 import envs
    from src_dreamerv3.envs import wrappers
    
    # Make a dummy env to extract spaces
    suite, task = config.task.split('_', 1)
    if suite == "dmc":
        from src_dreamerv3.envs import dmc
        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size,
            seed=config.seed
        )
        env = wrappers.NormalizeActions(env)
        env = wrappers.SelectAction(env, key="action")
    elif suite == "atari":
        from src_dreamerv3.envs import atari
        env = atari.Atari(
            task, config.action_repeat, config.size,
            gray=config.grayscale, noops=config.noops, lives=config.lives,
            sticky=config.stickey, actions=config.actions, resize=config.resize,
        )
        env = wrappers.OneHotAction(env)
        # CRITICAL FIX: Add SelectAction wrapper to unpack dict {'action': ...}
        env = wrappers.SelectAction(env, key="action")
    else:
        print(f"Unsupported suite for this simple script: {suite}")
        return

    obs_space = env.observation_space
    act_space = env.action_space
    
    config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    print(f"Action space size: {config.num_actions}")

    print("Reconstructing agent components...")
    
    # Shapes (assuming standard atari 64x64 grayscale/rgb)
    shapes = {
        'image': (1, 64, 64) if config.grayscale else (3, 64, 64),
        'vector': (0,), 
        'reward': (1,),
        'is_first': (1,),
        'is_last': (1,),
        'is_terminal': (1,),
        'action': (config.num_actions,), 
    }
    
    print(f"Obs shapes: {shapes}")
    
    # World Model
    wm = models.WorldModel(obs_space, act_space, 0, config).to(device)
    
    # Policy
    if config.use_director:
        task_behavior = hierarchical_policy.HierarchicalBehavior(
            config, wm
        ).to(device)
    else:
        print("Error: Config does not use Director.")
        return

    # Load checkpoint
    # Try latest.pt first (training state), then checkpoint.pt
    ckpt_path = logdir / 'latest.pt'
    if not ckpt_path.exists():
        ckpt_path = logdir / 'checkpoint.pt'
        
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    print(f"Loading checkpoint {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load state dicts
    # Checkpoint contains 'agent_state_dict' (from Dreamer._train items_to_save)
    if 'agent_state_dict' in checkpoint:
        agent_state = checkpoint['agent_state_dict']
        
        print("Checkpoint keys sample:", list(agent_state.keys())[:5])
        
        # Helper to clean keys
        def clean_keys(state_dict, prefix):
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    # Remove prefix (e.g. '_wm.')
                    new_key = k[len(prefix):]
                    # Remove '_orig_mod.' if present (from torch.compile)
                    new_key = new_key.replace('_orig_mod.', '')
                    new_state[new_key] = v
            return new_state

        # 1. Load World Model
        wm_state = clean_keys(agent_state, '_wm.')
        print(f"Found {len(wm_state)} keys for World Model")
        
        # Check if weights match before loading
        before_load = list(wm.parameters())[0].clone()
        
        keys = wm.load_state_dict(wm_state, strict=False)
        print(f"WM Load Results: missing={len(keys.missing_keys)}, unexpected={len(keys.unexpected_keys)}")
        if len(keys.missing_keys) > 0:
            print("Sample missing:", keys.missing_keys[:5])
            
        after_load = list(wm.parameters())[0]
        if torch.equal(before_load, after_load):
             print("WARNING: WM weights did not change! Loading likely failed.")
        else:
             print("WM weights updated successfully.")

        # 2. Load Task Behavior (includes Goal AE)
        tb_state = clean_keys(agent_state, '_task_behavior.')
        print(f"Found {len(tb_state)} keys for Task Behavior")
        
        keys = task_behavior.load_state_dict(tb_state, strict=False)
        print(f"TaskBehavior Load Results: missing={len(keys.missing_keys)}, unexpected={len(keys.unexpected_keys)}")
    else:
        print("Unknown checkpoint format")
        return

    print("Model loaded.")
    
    print("Collecting data for visualization...")
    obs = env.reset()
    
    # Store images
    real_images = []
    recon_images = []
    goal_images = [] 
    manager_goal_images = [] 
    
    action = torch.zeros(1, config.num_actions).to(device)
    state = None
    carry = None
    
    with torch.no_grad():
        for t in range(50): # Collect 50 steps
            # Preprocess obs
            obs = {k: np.array(v) for k, v in obs.items()}
            # Add batch dim
            obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
            
            # Normalize image to 0-1 (Dreamer expects this before running encoder which does x-0.5)
            # Typically env return 0-255 uint8.
            if 'image' in obs_tensor:
                 t = obs_tensor['image']
                 if t.dtype == torch.uint8:
                     t = t.float() / 255.0
                 obs_tensor['image'] = t
            
            # Embed
            embed = wm.encoder(obs_tensor)
            
            # Step World Model
            # obs_step(prev_state, prev_action, embed, is_first)
            is_first = obs_tensor['is_first'] if 'is_first' in obs_tensor else torch.tensor([[t==0]]).to(device)
            post, _ = wm.dynamics.obs_step(state, action, embed, is_first, sample=True)
            feat = wm.dynamics.get_feat(post)
            state = post
            
            # Run Goal AE visualization
            # Goal AE: feat -> z -> recon_feat
            # Note: Director GoalAE input is 'deter' part of feat usually, but check implementation
            # hierarchical_policy line 815: feat_size=self._goal_feat_size (deter only for RSSM)
            # We need to extract 'deter' part from feat if using RSSM
            
            # Check dynamics type
            dynamics_type = getattr(config, 'dynamics_type', 'rssm')
            if dynamics_type == 'rssm':
                # feat is concatenated [stoch, deter]. deter is usually the second part.
                # But wait, post['deter'] is the raw deterministic state.
                # hierarchical_policy passes goal_seq = feat[:, K:] where feat was post['deter']
                # So we should pass post['deter']
                ae_input = post['deter']
            else:
                # For VTA etc, might be different
                ae_input = post['deter'] # Assuming this is what we want
            
            g_recon_feat, z, _ = task_behavior.goal_ae(ae_input, sample=True)
            # Goal AE returns (z, dist) or (z, dist, metrics) depending on call?
            # task_behavior.goal_ae.forward returns (z, dist)
            # But line above implies 3 return values?
            # Wait, let's check HierarchicalBehavior.goal_ae call. 
            # It's an instance of GoalAutoencoder.
            # GoalAutoencoder.__call__ calls forward.
            # forward returns z, dist.
            # So `g_recon_feat` acts as first return value? No.
            # `g_recon_feat` variable name is misleading in my previous code if I thought it returned recon.
            # goal_ae.forward returns (z, dist).
            # z is the latent. dist is the distribution over features (reconstruction).
            # So:
            # Capturing all return values to avoid unpack error
            # GoalAutoencoder.forward returns (recon, z, dist)
            ret = task_behavior.goal_ae(ae_input, sample=True)
            recon_dist, z_sample = ret[0], ret[1]
            if hasattr(recon_dist, 'mode'):
                g_recon_feat = recon_dist.mode()
            else:
                g_recon_feat = recon_dist
            
            # Now we want to visualize this.
            # The World Model Decoder expects the FULL feature (stoch + deter).
            # We have 'deter' (reconstructed). We need 'stoch' to complete the picture.
            # We can use the current 'stoch' from 'post' combined with 'reconstructed deter'?
            # Or does Goal AE reconstruct everything? 
            # In Director, Goal AE usually reconstructs 'deter' part (Goal).
            
            # If Goal = Deter, then:
            # Reconstructed Full State = (stoch, Reconstructed Deter)
            # This is a hybrid state.
            
            # Let's try to decode:
            # 1. Original State (stoch, deter)
            full_feat = wm.dynamics.get_feat(post)
            # Decoder expects (Batch, Time, Dim), but we have (Batch, Dim). Add Time=1.
            full_feat = full_feat.unsqueeze(1)
            
            img_orig_dists = wm.heads['decoder'](full_feat)
            # FIXED: ConvDecoder already adds 0.5 (if sigmoid=False), so do NOT add 0.5 again.
            img_orig = img_orig_dists['image'].mode().cpu().numpy()
            # Output is (Batch, Time, H, W, C) -> (1, 1, 64, 64, 3)
            
            # 2. Reconstructed Goal State (stoch, RECON_deter)
            # Create a mock post state with replaced deter
            # Use wm.dynamics.get_feat to handle stoch/deter concatenation correctly
            fake_post = post.copy()
            fake_post['deter'] = g_recon_feat
            recon_full_feat = wm.dynamics.get_feat(fake_post)
            recon_full_feat = recon_full_feat.unsqueeze(1)
            
            img_goal_dists = wm.heads['decoder'](recon_full_feat)
            # FIXED: same here
            img_goal = img_goal_dists['image'].mode().cpu().numpy()
            
            # Save images
            
            real = obs['image'] # (C, H, W) or (H, W, C) depending on env wrapper
            if real.shape[0] in [1, 3]: real = real.transpose(1, 2, 0)
            real_images.append(np.clip(real/255.0, 0, 1))
            
            # img_orig[0, 0] is already (H, W, C) from ConvDecoder
            recon_images.append(np.clip(img_orig[0, 0], 0, 1))
            goal_images.append(np.clip(img_goal[0, 0], 0, 1))
            
            # Step Action
            if carry is None:
                carry = task_behavior.initial(1)
                
            outs, carry = task_behavior.policy(post, carry)

            # --- 4. Manager Goal (Planned Deter) ---
            # carry['goal'] is the active goal (deter)
            mgr_goal = carry['goal']
            
            # Create mock post with Manager's goal
            mgr_post = post.copy()
            mgr_post['deter'] = mgr_goal
            mgr_full_feat = wm.dynamics.get_feat(mgr_post)
            mgr_full_feat = mgr_full_feat.unsqueeze(1)
            
            img_mgr_dists = wm.heads['decoder'](mgr_full_feat)
            img_mgr = img_mgr_dists['image'].mode().cpu().numpy()
            manager_goal_images.append(np.clip(img_mgr[0, 0], 0, 1))
            # ---------------------------------------

            action_tensor = outs['action'].sample()
            
            # Ensure action is one-hot
            # If action is already one-hot, shape will be (B, num_actions)
            # If not (e.g. indices), shape will be (B,) or (B, 1)
            # Wrapper expects (num_actions,) numpy array
            
            action_np = action_tensor[0].cpu().detach().numpy()
            
            if action_np.shape == () or action_np.shape == (1,):
                # Scalar index -> convert to one-hot matching env wrapper expectation
                idx = int(action_np)
                onehot = np.zeros(config.num_actions, dtype=np.float32)
                onehot[idx] = 1.0
                action_to_env = onehot
            elif action_np.shape == (config.num_actions,):
                # Already one-hot vector
                action_to_env = action_np
            else:
                # Unexpected shape
                print(f"Warning: Unexpected action shape {action_np.shape}. Trying to infer.")
                action_to_env = action_np

            obs = env.step({'action': action_to_env})
            if isinstance(obs, tuple):
                obs = obs[0]
                
            # Update action for next step loop (expecting tensor)
            if action_to_env.shape == (config.num_actions,):
                action = torch.from_numpy(action_to_env).unsqueeze(0).to(device)
            else:
                action = action_tensor

    env.close()
    
    # Plotting
    print("Generating visualization plot...")
    
    indices = range(0, 50, 5)
    num_frames = len(indices)
    
    fig, axes = plt.subplots(4, num_frames, figsize=(num_frames*2, 8))
    
    for i, idx in enumerate(indices):
        # Row 1: Real Image
        axes[0, i].imshow(real_images[idx], cmap='gray' if real_images[idx].shape[-1]==1 else None)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Real Env")
        
        # Row 2: WM Recon
        axes[1, i].imshow(recon_images[idx], cmap='gray' if recon_images[idx].shape[-1]==1 else None)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("WM Recon")
        
        # Row 3: Goal AE Recon
        axes[2, i].imshow(goal_images[idx], cmap='gray' if goal_images[idx].shape[-1]==1 else None)
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Goal AE Recon")
        
        # Row 4: Manager Goal
        axes[3, i].imshow(manager_goal_images[idx], cmap='gray' if manager_goal_images[idx].shape[-1]==1 else None)
        axes[3, i].axis('off')
        if i == 0: axes[3, i].set_title("Manager Goal")

    plt.tight_layout()
    save_path = logdir / 'goal_ae_vis.png'
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == '__main__':
    main()
