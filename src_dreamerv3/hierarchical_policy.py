"""
Hierarchical Policy for Director-style Learning with VTA

Based on "Deep Hierarchical Planning from Pixels" (Director)
https://arxiv.org/abs/2206.04114

This module implements:
- Manager Policy: Selects subgoals (skills) from abstract states
- Worker Policy: Executes primitive actions to achieve subgoals
- Hierarchical Behavior: Coordinates training of both policies

Key difference from Director:
- Supports VTA boundary-driven manager calls (variable K steps)
- Manager takes abs_feat (temporal abstraction) as input
- Worker takes obs_feat + goal as input
"""

import copy
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from . import networks
from . import tools
from .goal_autoencoder import GoalAutoencoder, GoalEncoder, GoalDecoder


class MultiCategoricalDist:
    """
    Multi-variable categorical distribution for Manager skills.
    
    Wraps multiple independent OneHotDist distributions.
    This is the correct representation for (stoch, discrete) skill space
    where each of the 'stoch' variables independently selects from 'discrete' categories.
    
    Official Director uses this structure: entropy = stoch * log(discrete) ≈ 16.6
    (not log(stoch * discrete) ≈ 4.16 which is wrong)
    """
    
    def __init__(self, logits, stoch, discrete, unimix_ratio=0.01):
        """
        Args:
            logits: Raw logits of shape (..., stoch * discrete)
            stoch: Number of categorical variables
            discrete: Number of categories per variable
            unimix_ratio: Uniform mixing ratio for exploration
        """
        self._stoch = stoch
        self._discrete = discrete
        
        # Reshape to (..., stoch, discrete)
        self._logits = logits.reshape(logits.shape[:-1] + (stoch, discrete))
        
        # Create OneHotDist for each categorical variable
        self._dist = tools.OneHotDist(self._logits, unimix_ratio=unimix_ratio)
        
        # For entropy calculation bounds
        self.minent = 0.0
        self.maxent = float(stoch * torch.log(torch.tensor(discrete, dtype=torch.float32)))
    
    @property
    def logits(self):
        return self._logits
    
    def sample(self, sample_shape=()):
        """Sample from all categorical variables independently."""
        return self._dist.sample(sample_shape)
    
    def mode(self):
        """Return mode of each categorical variable."""
        return self._dist.mode()
    
    def log_prob(self, value):
        """
        Compute log probability, summing over all categorical variables.
        
        Args:
            value: One-hot encoded value of shape (..., stoch, discrete)
            
        Returns:
            Log probability of shape (...)
        """
        # Ensure value has correct shape
        if value.shape[-2:] != (self._stoch, self._discrete):
            value = value.reshape(value.shape[:-1] + (self._stoch, self._discrete))
        
        # Sum log_prob over stoch dimension (independent variables)
        per_var_log_prob = self._dist.log_prob(value)  # (..., stoch)
        return per_var_log_prob.sum(dim=-1)  # (...)
    
    def entropy(self):
        """
        Compute entropy, summing over all categorical variables.
        
        Returns:
            Entropy of shape (...)
        """
        per_var_entropy = self._dist.entropy()  # (..., stoch)
        return per_var_entropy.sum(dim=-1)  # (...)


to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """Running mean and std for reward normalization."""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class AutoEntropyAdjust:
    """
    Automatic entropy adjustment (like official Director's actent).
    
    Adjusts the entropy bonus scale to maintain a target entropy level.
    Uses multiplicative adjustment for smooth control.
    """
    
    def __init__(self, scale=3e-3, target=0.5, min_scale=1e-5, max_scale=1e2, velocity=0.1, device=None, perdim=False, shape=()):
        """
        Args:
            scale: Initial entropy bonus scale
            target: Target entropy value to maintain
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
            velocity: Adjustment velocity (higher = faster adjustment)
            device: Device to use
            perdim: If True, maintain per-dimension entropy scales (official: actent_perdim)
            shape: Shape for per-dimension scales (e.g., (stoch,) for Manager)
        """
        self._target = target
        self._min = min_scale
        self._max = max_scale
        self._velocity = velocity
        self._device = device
        self._perdim = perdim
        
        # Official: actent_perdim: True
        if perdim and len(shape) > 0:
            self._scale = torch.full(shape, scale, dtype=torch.float32, device=device)
        else:
            self._scale = scale
    
    def __call__(self, entropy):
        """
        Compute entropy bonus and adjust scale.
        
        Args:
            entropy: Current entropy (scalar or per-dimension tensor)
            
        Returns:
            scale: Current entropy bonus scale (to multiply with entropy)
        """
        # Compute adjustment factor based on error
        # Note: Per-dimension adjustment requires entropy to have shape (..., stoch)
        # If entropy is flattened (T*B,), fall back to scalar adjustment
        if (self._perdim and isinstance(self._scale, torch.Tensor) and 
            isinstance(entropy, torch.Tensor) and entropy.dim() > 1 and
            entropy.shape[-1] == self._scale.shape[0]):
            # Per-dimension adjustment when shapes align
            error = self._target - entropy.detach().mean(dim=tuple(range(entropy.dim()-1)))  # Mean over batch dims
            # Official: scale *= exp(rate * error)
            adjustment = torch.exp(self._velocity * error)
            self._scale = torch.clamp(self._scale * adjustment, self._min, self._max)
            return self._scale
        else:
            # Scalar adjustment (default for most cases)
            error = self._target - entropy.detach().mean()
            # Official: scale *= exp(rate * error)
            adjustment = torch.exp(torch.tensor(self._velocity * error.item()))
            if isinstance(self._scale, torch.Tensor):
                self._scale = self._scale.mean().item()
            self._scale = max(self._min, min(self._max, self._scale * adjustment.item()))
            return self._scale
    
    @property
    def scale(self):
        if isinstance(self._scale, torch.Tensor):
            return self._scale.mean().item()
        return self._scale


class VFunction(nn.Module):
    """
    Value Function matching official Director's implementation.
    
    Key differences from simple Value Network:
    - Uses distribution output (symlog_disc)
    - Loss computed via log_prob, not MSE
    - Supports slow target network
    - Computes GAE/GVE returns
    """
    
    def __init__(self, input_size, layers, units, act, norm, dist, 
                 outscale, device, config, reward_fn=None, name="VFunction"):
        super().__init__()
        self._config = config
        self._reward_fn = reward_fn
        self._name = name
        self._device = device
        
        # Main network
        self.net = networks.MLP(
            input_size,
            (255,) if dist == "symlog_disc" else (),
            layers,
            units,
            act,
            norm,
            dist,
            outscale=outscale,
            device=device,
            name=f"{name}_net",
        )
        
        # Slow target network
        if config.critic["slow_target"]:
            self.target_net = networks.MLP(
                input_size,
                (255,) if dist == "symlog_disc" else (),
                layers,
                units,
                act,
                norm,
                dist,
                outscale=outscale,
                device=device,
                name=f"{name}_target",
            )
            # Copy initial weights
            # Copy initial weights
            # MLP uses name prefix for layers, so keys must be renamed
            net_state = self.net.state_dict()
            target_state = {}
            prefix_from = f"{name}_net"
            prefix_to = f"{name}_target"
            
            for k, v in net_state.items():
                new_key = k.replace(prefix_from, prefix_to)
                if new_key not in self.target_net.state_dict():
                    # Fallback for keys that don't depend on name (if any) or unexpected structure
                    print(f"Warning: Key {new_key} not found in target net. Using original {k}")
                    new_key = k
                target_state[new_key] = v
            
            self.target_net.load_state_dict(target_state)
            # Freeze target network
            for param in self.target_net.parameters():
                param.requires_grad = False
            self._updates = 0
        else:
            self.target_net = self.net
            self._updates = 0
    
    def forward(self, inputs):
        """Return distribution from main network."""
        return self.net(inputs)
    
    def target(self, inputs):
        """Return distribution from target network."""
        return self.target_net(inputs)
    
    def compute_target(self, features, reward, discount, impl='gve'):
        """
        Compute TD-lambda returns (GAE or GVE).
        
        Args:
            features: Input features for value network (T, B, F)
            reward: Reward tensor (T, B)
            discount: Discount tensor (T, B)
            impl: 'gae' or 'gve'
            
        Returns:
            target: TD-lambda target (T, B)
            baseline: Value baseline (T, B)
        """
        lambda_ = self._config.discount_lambda
        
        # Get value from target network using features (Tensor)
        value = self.target_net(features).mode()
        if value.dim() > 2:
            value = value.squeeze(-1)
        
        # Ensure discount matches reward shape
        disc = discount
        T = reward.shape[0]
        if disc.shape[0] > T:
            disc = disc[:T]
        
        if impl == 'gae':
            # Generalized Advantage Estimation
            advs = [torch.zeros_like(value[0])]
            deltas = reward + disc * value[1:T+1] - value[:T]
            for t in reversed(range(T)):
                advs.append(deltas[t] + disc[t] * lambda_ * advs[-1])
            adv = torch.stack(list(reversed(advs))[:-1])
            return adv + value[:T], value[:T]
            
        elif impl == 'gve':
            # Generalized Value Estimation (default in Director)
            vals = [value[-1]]
            interm = reward + disc * value[1:T+1] * (1 - lambda_)
            for t in reversed(range(T)):
                vals.append(interm[t] + disc[t] * lambda_ * vals[-1])
            ret = torch.stack(list(reversed(vals))[:-1])
            return ret, value[:T]
        else:
            raise NotImplementedError(impl)
    
    def compute_target_with_disc(self, features, reward, disc, impl='gve'):
        """
        Compute TD-lambda returns with pre-computed disc (Official Director style).
        
        This method takes disc directly (already computed as cont[1:] * discount)
        to avoid dimension mismatch issues from split_traj.
        
        Args:
            features: Input features (T+1, B, F)
            reward: Reward tensor (T, B)
            disc: Pre-computed discount = cont * discount_scalar (T, B)
            impl: 'gae' or 'gve'
        """
        lambda_ = self._config.discount_lambda
        
        # Get value from target network
        value = self.target_net(features).mode()
        if value.dim() > 2:
            value = value.squeeze(-1)
        
        T = reward.shape[0]
        
        if impl == 'gae':
            # Generalized Advantage Estimation
            advs = [torch.zeros_like(value[0])]
            deltas = reward + disc * value[1:T+1] - value[:T]
            for t in reversed(range(T)):
                advs.append(deltas[t] + disc[t] * lambda_ * advs[-1])
            adv = torch.stack(list(reversed(advs))[:-1])
            return adv + value[:T], value[:T]
            
        elif impl == 'gve':
            # Generalized Value Estimation (default in Director)
            vals = [value[-1]]
            interm = reward + disc * value[1:T+1] * (1 - lambda_)
            for t in reversed(range(T)):
                vals.append(interm[t] + disc[t] * lambda_ * vals[-1])
            ret = torch.stack(list(reversed(vals))[:-1])
            return ret, value[:T]
        else:
            raise NotImplementedError(impl)
    
    def loss(self, features, target, weight):
        """
        Compute log_prob loss (matching official Director).
        
        Args:
            features: Input features for value network
            target: TD-lambda target values
            weight: Discount weight for loss weighting
            
        Returns:
            loss: Scalar loss
            metrics: Dict of metrics
        """
        # Slice features to match target length if necessary
        T = target.shape[0]
        if features.shape[0] > T:
            features_sliced = features[:T]
        else:
            features_sliced = features
            
        dist = self.net(features_sliced)
        
        # Official Director: loss = -(dist.log_prob(target) * weight).mean()
        log_prob = dist.log_prob(target.detach())
        loss = -(log_prob * weight.detach()).mean()
        
        metrics = {
            f'{self._name}_loss': loss.detach().cpu(),
            f'{self._name}_mean': dist.mode().mean().detach().cpu(),
        }
        
        return loss, metrics
    
    def update_slow(self):
        """Update slow target network."""
        if not self._config.critic["slow_target"]:
            return
        
        if self._updates >= self._config.critic["slow_target_update"]:
            self._updates = 0
            mix = self._config.critic["slow_target_fraction"]
            with torch.no_grad():
                for s, d in zip(self.net.parameters(), self.target_net.parameters()):
                    d.data.copy_(mix * s.data + (1 - mix) * d.data)
        
        self._updates += 1


class ImagActorCritic(nn.Module):
    """
    Actor-Critic matching official Director's implementation.
    
    Features:
    - Multiple VFunctions for different reward types (extr, expl, goal)
    - Score aggregation with weights
    - Entropy normalization
    - REINFORCE or backprop actor gradient
    """
    
    def __init__(self, vfns, scales, actor_net, act_space, config, device, name="ImagActorCritic", skill_shape=None, actor_grad=None):
        """
        Args:
            vfns: Dict of {name: VFunction} for each reward type
            scales: Dict of {name: scale} for score weighting
            actor_net: Actor network module
            act_space: Action space
            config: Config object
            config: Config object
            device: Device
            name: Name for logging
            skill_shape: Tuple (stoch, discrete) for Manager multi-variable distribution, None for Worker
            actor_grad: 'reinforce' or 'dynamics' (default: None, use config default)
        """
        super().__init__()
        self._config = config
        self._device = device
        self._name = name
        self._scales = scales
        self._skill_shape = skill_shape
        self._actor_grad = actor_grad
        
        # Determine gradient type
        if self._actor_grad is None:
            if hasattr(config, 'actor_grad'):
                self._actor_grad = config.actor_grad
            else:
                self._actor_grad = 'dynamics' # Default for DreamerV3
        
        # Store VFunctions as ModuleDict for proper parameter tracking
        self.vfns = nn.ModuleDict(vfns)
        self.actor = actor_net
        
        self._act_space = act_space
        
        # Reward EMA for each VFunction
        self.retnorms = {}
        self.scorenorms = {}
        for key in vfns:
            self.retnorms[key] = RewardEMA(device)
            self.scorenorms[key] = RewardEMA(device)
        
        # Advantage normalizer
        self.advnorm = RewardEMA(device)
        self.adv_ema_vals = torch.zeros(2, device=device)
        
        # Entropy adjustment with normalization
        # Official: actent_perdim: True for multi-dimensional actions
        perdim = getattr(config.actor, "actent_perdim", False) or name == "Manager"
        ent_shape = act_space.shape[:-1] if (hasattr(act_space, 'discrete') and act_space.discrete) or (hasattr(act_space, 'shape') and len(act_space.shape) >= 2) else ()
        
        self._actent = AutoEntropyAdjust(
            scale=config.actor.get("actent_scale", 3e-3),
            target=config.actor.get("actent_target", 0.5),
            min_scale=1e-5,
            max_scale=1e2,
            velocity=0.1,
            device=device,
            perdim=perdim,
            shape=ent_shape,
        )
        self._actent_norm = config.actor.get("actent_norm", True)
        
        # EMA values for return normalization
        self._ret_ema_vals = {k: torch.zeros(2, device=device) for k in vfns}
        self._score_ema_vals = {k: torch.zeros(2, device=device) for k in vfns}
    
    def forward(self, inputs):
        """
        Forward pass for policy network.
        For Manager: wraps raw logits with MultiCategoricalDist
        For Worker: delegates to actor network directly
        """
        output = self.actor(inputs)
        
        # If this is Manager (has skill_shape), wrap with MultiCategoricalDist
        if self._skill_shape is not None:
            stoch, discrete = self._skill_shape
            return MultiCategoricalDist(output, stoch, discrete, unimix_ratio=0.01)
        
        return output
    
    def score(self, features, traj, weight):
        """
        Compute aggregated score for actor training.
        
        Args:
            features: Input features for value networks (T, B, F)
            traj: Trajectory dict with rewards/discounts
            weight: Discount weight
            
        Returns:
            score: Aggregated score (T, B)
        """
        scores = []
        discount_scalar = self._config.discount
        
        for key, vfn in self.vfns.items():
            if self._scales.get(key, 0.0) == 0.0:
                continue
                
            reward_key = f'reward_{key}'
            if reward_key not in traj:
                continue
                
            reward = traj[reward_key]
            
            # Compute disc from cont like official Director (matching update method)
            T = reward.shape[0]
            expected_batch = reward.shape[1] if reward.dim() > 1 else 1
            
            if 'cont' in traj:
                cont = traj['cont']
                if cont.shape[0] > T:
                    disc = cont[1:T+1] * discount_scalar
                else:
                    disc = cont * discount_scalar
                    
                # Verify batch dimension matches reward
                disc_batch = disc.shape[1] if disc.dim() > 1 else 1
                if disc_batch != expected_batch:
                    disc = discount_scalar * torch.ones_like(reward)
            else:
                disc = discount_scalar * torch.ones_like(reward)

            # Squeeze dim -1 if disc is (T, B, 1)
            if disc.dim() > reward.dim():
                disc = disc.squeeze(-1)
            
            # Compute return and baseline using features
            ret, baseline = vfn.compute_target_with_disc(
                features, reward, disc,
                impl=self._config.critic.get("return", "gve")
            )
            
            # Normalize return
            offset, scale = self.retnorms[key](ret.flatten(), self._ret_ema_vals[key])
            norm_ret = (ret - offset) / scale
            norm_baseline = (baseline - offset) / scale
            
            # Score = normalized return - baseline
            score = norm_ret - norm_baseline
            
            # Official: scorenorm: {impl: off} - DISABLED
            # offset, scale = self.scorenorms[key](score.flatten(), self._score_ema_vals[key])
            # score = (score - offset) / scale
            
            scores.append(score * self._scales[key])
        
        if not scores:
            return torch.zeros((T, weight.shape[1]), device=weight.device)
        
        # Aggregate scores
        total_score = torch.stack(scores, dim=0).sum(dim=0)
        
        # Normalize aggregated advantage
        offset, scale = self.advnorm(total_score.flatten(), self.adv_ema_vals)
        total_score = (total_score - offset) / scale
        
        return total_score
    
    def actor_loss(self, features, traj, score, weight):
        """
        Compute actor loss with entropy bonus.
        
        Args:
            features: Input features for actor (T, B, F)
            traj: Trajectory (for actions)
            score: Aggregated score
            weight: Discount weight
            
        Returns:
            loss: Scalar loss
            metrics: Dict of metrics
        """
        metrics = {}
        
        # Get policy distribution using features
        # Slice features to match action length
        action = traj['action']
        T = action.shape[0]
        
        if features.shape[0] > T:
            features_sliced = features[:T]
        else:
            features_sliced = features
            
        # Official: https://github.com/danijar/director/blob/main/embodied/agents/director/agent.py#L134
        # policy = self.actor(tf.stop_gradient(features))
        if self._actor_grad == 'reinforce':
            # Detach features for REINFORCE to avoid backprop through dynamics/trajectory
            # This prevents "inplace operation" errors when Worker updates parameters used in trajectory generation
            policy = self.forward(features_sliced.detach()) # Use forward() for proper distribution wrapping
        else:
            policy = self.forward(features_sliced)  # Use forward() for proper distribution wrapping
        
        # REINFORCE loss
        log_prob = policy.log_prob(action.detach())
        T_score = score.shape[0]  # Use score shape as authoritative length
        
        # Official: loss = -policy.log_prob(action)[:-1] * tf.stop_gradient(score)
        # Slice log_prob to match score length
        if log_prob.shape[0] > T_score:
            log_prob = log_prob[:T_score]
            
        actor_loss = -(log_prob * score.detach()) * weight[:T_score].detach()
        
        # Entropy with normalization (matching official Director)
        ent = policy.entropy()[:T_score]
        
        if self._actent_norm:
            # Normalize entropy to [0, 1] range
            # For discrete distributions: max_ent = log(num_actions)
            if hasattr(policy, 'maxent') and hasattr(policy, 'minent'):
                lo, hi = policy.minent, policy.maxent
            else:
                # Estimate from action space
                shape = self._act_space.shape
                # For Manager (stoch, discrete), max ent is stoch * log(discrete)
                # For Worker (discrete actions), max ent is log(num_actions)
                if len(shape) >= 2:
                    # Multi-dimensional discrete (e.g. VAE latents)
                    stoch, discrete = shape[-2], shape[-1]
                    # Assuming independent discrete variables
                    hi = float(stoch * torch.log(torch.tensor(discrete, dtype=torch.float32)))
                else:
                    # Single discrete action
                    hi = float(torch.log(torch.tensor(shape[-1], dtype=torch.float32)))
                lo = 0.0
            ent_norm = (ent - lo) / (hi - lo + 1e-8)
        else:
            ent_norm = ent
        
        # Get entropy scale (adjusts to maintain target)
        ent_scale = self._actent(ent_norm)
        
        # Entropy loss
        ent_loss = -ent_scale * ent_norm
        
        # Combined loss
        loss = (actor_loss + ent_loss).mean()
        
        metrics[f'{self._name}_actor_loss'] = actor_loss.mean().detach().cpu()
        metrics[f'{self._name}_entropy'] = ent.mean().detach().cpu()
        metrics[f'{self._name}_entropy_scale'] = ent_scale.detach().cpu() if torch.is_tensor(ent_scale) else torch.tensor(ent_scale).cpu()
        
        return loss, metrics
    
    def update(self, features, traj, weight):
        """
        Full update for actor and all critics.
        
        Args:
            features: Input features for actor/critic (T+1, B, F)
            traj: Trajectory with all reward types and cont
            weight: Discount weight
            
        Returns:
            total_loss: Combined loss for backprop
            metrics: Dict of metrics
        """
        metrics = {}
        discount_scalar = self._config.discount
        
        # Normalize weight shape (remove last dim 1 if present) to match target/log_prob
        if weight.dim() > 2:
            weight = weight.squeeze(-1)

        # Detach features for REINFORCE (Manager) to prevent backprop through trajectory/dynamics
        # This solves the "inplace operation" error by isolating Manager optimization from Worker/WM
        if self._actor_grad == 'reinforce':
            features = features.detach()
        
        # Update each VFunction
        total_vfn_loss = 0.0
        for key, vfn in self.vfns.items():
            if self._scales.get(key, 0.0) == 0.0:
                continue
                
            reward_key = f'reward_{key}'
            if reward_key not in traj:
                continue
            
            reward = traj[reward_key]
            
            # Compute disc from cont like official Director: disc = traj['cont'][1:] * discount
            # Ensure disc shape matches reward shape (authoritative source)
            T = reward.shape[0]
            expected_batch = reward.shape[1] if reward.dim() > 1 else 1
            
            if 'cont' in traj:
                cont = traj['cont']
                if cont.shape[0] > T:
                    disc = cont[1:T+1] * discount_scalar
                else:
                    disc = cont * discount_scalar
                    
                # Verify batch dimension matches reward
                disc_batch = disc.shape[1] if disc.dim() > 1 else 1
                if disc_batch != expected_batch:
                    # Fallback: create disc with correct shape from reward
                    disc = discount_scalar * torch.ones_like(reward)
            else:
                disc = discount_scalar * torch.ones_like(reward)
            
            # Squeeze dim -1 if disc is (T, B, 1) to match (T, B) target/value
            if disc.dim() > reward.dim():
                disc = disc.squeeze(-1)

            # Compute target (pass disc directly, not separate discount)
            target, _ = vfn.compute_target_with_disc(
                features, reward, disc,
                impl=self._config.critic.get("return", "gve")
            )
            
            # Compute loss
            loss, mets = vfn.loss(features, target, weight[:-1])
            total_vfn_loss = total_vfn_loss + loss
            
            metrics.update({f'{self._name}_{key}_{k}': v for k, v in mets.items()})
            
            # Update target network
            vfn.update_slow()
        
        # Compute score for actor
        score = self.score(features, traj, weight)
        
        # Actor loss
        actor_loss, actor_mets = self.actor_loss(features, traj, score, weight)
        metrics.update(actor_mets)
        
        total_loss = actor_loss + total_vfn_loss
        
        return total_loss, metrics


class HierarchicalBehavior(nn.Module):
    """
    Hierarchical behavior with Manager and Worker policies.
    
    Manager: Selects subgoals in discrete latent space
    Worker: Executes primitive actions to achieve subgoals
    """
    
    def __init__(self, config, world_model):
        super().__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        
        # Dynamics type and feature sizes
        dynamics_type = getattr(config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            self._obs_feat_size = world_model.dynamics._obs_feat_size
            if getattr(config, 'director_manager_input', 'abs_feat') == 'obs_feat':
                self._abs_feat_size = self._obs_feat_size
            else:
                self._abs_feat_size = world_model.dynamics._abs_feat_size
            self._feat_size = world_model.dynamics.feat_size  # obs + abs
        else:
            if config.dyn_discrete:
                self._feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            else:
                self._feat_size = config.dyn_stoch + config.dyn_deter
            self._obs_feat_size = self._feat_size
            self._abs_feat_size = self._feat_size
        
        # Skill/goal configuration
        self._skill_stoch = config.director_skill_stoch
        self._skill_discrete = config.director_skill_discrete
        self._skill_size = self._skill_stoch * self._skill_discrete
        
        # Whether to use delta (goal - current) or absolute goal
        self._use_delta = config.director_use_delta
        
        # Manager timing mode
        self._manager_mode = getattr(config, 'director_manager_mode', 'fixed')  # 'fixed' or 'boundary'
        self._skill_duration = config.director_skill_duration
        
        # Context size for conditional goal decoding (using belief/deter like official Director)
        dynamics_type = getattr(config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            self._goal_context_size = config.vta_obs_belief  # obs_belief size
            self._goal_feat_size = config.vta_obs_belief  # Goal space = deter only (like official)
        else:
            self._goal_context_size = config.dyn_deter  # deter size
            self._goal_feat_size = config.dyn_deter  # Goal space = deter only
        
        # Goal Autoencoder (encodes goal_feat to discrete skill)
        # CRITICAL: Uses goal_feat_size (deter only), NOT obs_feat_size
        self.goal_ae = GoalAutoencoder(
            feat_size=self._goal_feat_size,  # Only deterministic state
            hidden=config.director_goal_ae_hidden,
            layers=config.director_goal_ae_layers,
            stoch=self._skill_stoch,
            discrete=self._skill_discrete,
            kl_config=config.director_goal_ae_kl_config,
            unimix_ratio=config.director_goal_ae_unimix,
            context_size=0,  # CRITICAL: Context not used in official Director for AE
            device=config.device,
            dist=getattr(config, 'goal_ae_dist', 'mse'),
        )
        
        # Reward weights
        self._manager_extr_weight = config.director_manager_extr_weight
        self._manager_expl_weight = config.director_manager_expl_weight
        # Worker reward weights (like official Director worker_rews)
        self._worker_extr_weight = getattr(config, 'director_worker_extr_weight', 0.0)
        self._worker_goal_weight = config.director_worker_goal_weight
        
        
        # Goal reward type
        self._goal_reward_type = config.director_goal_reward
        # Worker input size
        # obs_feat + goal + [delta] + [abs_feat]
        # For RSSM, abs_feat is redundant (same as obs_feat), so we exclude it to match official Director
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        
        self._worker_input_size = self._obs_feat_size + self._goal_feat_size
        
        if self._use_delta:
             self._worker_input_size += self._goal_feat_size
             
        if dynamics_type != 'rssm':
             self._worker_input_size += self._abs_feat_size
        
        
        # === MANAGER ACTOR-CRITIC (Official Director Structure) ===
        # Manager learns from extrinsic + exploration rewards
        
        # 1. Manager Actor Net - Custom network outputting raw logits
        # Using shape=None makes MLP return hidden layer output, then we add our own output layer
        class ManagerActorNet(nn.Module):
            """Manager actor that outputs raw logits for multi-variable distribution."""
            def __init__(self, inp_size, out_size, layers, units, act, norm, outscale, device):
                super().__init__()
                act_fn = getattr(torch.nn, act)
                
                # Build hidden layers
                hidden_layers = []
                inp = inp_size
                for i in range(layers):
                    hidden_layers.append(nn.Linear(inp, units, bias=not norm))
                    if norm:
                        hidden_layers.append(nn.LayerNorm(units, eps=1e-03))
                    hidden_layers.append(act_fn())
                    inp = units
                self.hidden = nn.Sequential(*hidden_layers)
                self.hidden.apply(tools.weight_init)
                
                # Output layer (raw logits)
                self.output = nn.Linear(units, out_size)
                self.output.apply(tools.uniform_weight_init(outscale))
            
            def forward(self, x):
                h = self.hidden(x)
                return self.output(h)  # Raw logits, no distribution
        
        manager_net = ManagerActorNet(
            self._abs_feat_size,
            self._skill_stoch * self._skill_discrete,
            config.director_manager_layers,
            config.director_manager_units,
            config.act,
            config.norm,
            config.director_manager_outscale,
            config.device,
        )
        
        # 2. Manager VFunctions
        # Official: {extr: VFunction, expl: VFunction, goal: VFunction}
        manager_vfns = {
            "extr": VFunction(
                self._abs_feat_size, config.director_manager_layers, config.director_manager_units,
                config.act, config.norm, config.critic["dist"], config.critic["outscale"],
                config.device, config, name="ManagerVOpt_Extr"
            ),
            "expl": VFunction(
                self._abs_feat_size, config.director_manager_layers, config.director_manager_units,
                config.act, config.norm, config.critic["dist"], config.critic["outscale"],
                config.device, config, name="ManagerVOpt_Expl"
            ),
        }
        
        # 3. Manager Reward Scales
        manager_scales = {
            "extr": self._manager_extr_weight,
            "expl": self._manager_expl_weight,
            "goal": 0.0,
        }
        
        # 4. ImagActorCritic
        # Define FakeSpace for Manager (stoch, discrete) for entropy calc
        class FakeSpace:
            def __init__(self, shape): self.shape = shape
            
        # Official Director: Manager uses SAME config as Worker (NOT gamma^K)
        # The temporal abstraction is handled by abstract_traj, not by changing discount
            
        self.manager = ImagActorCritic(
            manager_vfns, manager_scales, manager_net, 
            FakeSpace((self._skill_stoch, self._skill_discrete)), # Pass skill space shape
            config, config.device, name="Manager",
            skill_shape=(self._skill_stoch, self._skill_discrete),  # CRITICAL FIX: Enable multi-variable distribution
            actor_grad='reinforce',  # Manager official: reinforce
        )
        
        
        # === WORKER ACTOR-CRITIC (Official Director Structure) ===
        # Worker learns from goal + extrinsic rewards
        
        # 1. Worker Actor Net
        worker_net = networks.MLP(
            self._worker_input_size,
            (config.num_actions,),
            config.director_worker_layers,
            config.director_worker_units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.director_worker_outscale,
            name="WorkerInfo",
        )
        
        # 2. Worker VFunctions
        worker_vfns = {
            "goal": VFunction(
                self._worker_input_size, config.director_worker_layers, config.director_worker_units,
                config.act, config.norm, config.critic["dist"], config.critic["outscale"],
                config.device, config, name="WorkerVOpt_Goal"
            ),
            "extr": VFunction(
                self._worker_input_size, config.director_worker_layers, config.director_worker_units,
                config.act, config.norm, config.critic["dist"], config.critic["outscale"],
                config.device, config, name="WorkerVOpt_Extr"
            ),
        }
        
        # 3. Worker Reward Scales
        worker_scales = {
            "goal": self._worker_goal_weight,
            "extr": self._worker_extr_weight,
            "expl": 0.0,
        }
        
        # 4. ImagActorCritic
        # Need fake action space for entropy estimation fallback
        class FakeSpace:
            def __init__(self, shape): self.shape = shape
        
        self.worker = ImagActorCritic(
            worker_vfns, worker_scales, worker_net,
            FakeSpace((config.num_actions,)),
            config, config.device, name="Worker"
        )
        
        # Remove standalone values (they are inside ImagActorCritic now)
        # self.manager_value = ... (Removed)
        # self.worker_value = ... (Removed)
        
        # Slow target networks
        
        # Slow target networks logic is now handled inside VFunction/ImagActorCritic

        
        # Optimizers
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        
        self._goal_ae_opt = tools.Optimizer(
            "goal_ae",
            self.goal_ae.parameters(),
            config.director_goal_ae_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        
        self._manager_opt = tools.Optimizer(
            "manager",
            self.manager.parameters(),
            config.director_manager_lr,
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        
        self._worker_opt = tools.Optimizer(
            "worker",
            self.worker.parameters(),
            config.director_worker_lr,
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        
        # Combined optimizer for joint training (worker + manager)
        # Using parameters() from ImagActorCritic includes both actor and value networks
        self._combined_opt = torch.optim.Adam(
            list(self.manager.parameters()) + list(self.worker.parameters()),
            lr=config.director_worker_lr,  # Use worker LR as baseline
            eps=config.actor["eps"],
        )
        self._combined_scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)
        
        # Reward normalization
        if self._config.reward_EMA:
            self.register_buffer("manager_ema_vals", torch.zeros((2,), device=config.device))
            self.register_buffer("worker_ema_vals", torch.zeros((2,), device=config.device))
            self.manager_reward_ema = RewardEMA(device=config.device)
            self.worker_reward_ema = RewardEMA(config.device)
        self.worker_ema_vals = torch.zeros(2, device=config.device)
        
        # Automatic Entropy Adjustment
        self.worker_entropy_adj = AutoEntropyAdjust(
            scale=getattr(config, 'director_worker_entropy_scale', 3e-3),
            target=getattr(config, 'director_worker_entropy_target', 0.5),
            min_scale=getattr(config, 'director_worker_entropy_min', 1e-5),
            max_scale=getattr(config, 'director_worker_entropy_max', 1e2),
            velocity=getattr(config, 'director_worker_entropy_vel', 0.1),
            device=config.device,
        )
        
        # Metrics
        self._metrics = {}
        self._update_count = 0  # Counter for periodic logging
        
        # Reward weights and Goal reward type moved to earlier initialization

        
        print(f"HierarchicalBehavior initialized:")
        print(f"  Manager input: {self._abs_feat_size}, output: {self._skill_size}")
        print(f"  Worker input: {self._worker_input_size}, output: {config.num_actions}")
        print(f"  Goal space: {self._goal_feat_size} (deterministic only)")
        print(f"  Worker rewards: goal={self._worker_goal_weight}, extr={self._worker_extr_weight}")
        print(f"  Manager mode: {self._manager_mode}, duration: {self._skill_duration}")

    def initial(self, batch_size):
        """Initialize hierarchical state for policy rollout."""
        return {
            "step": torch.zeros((batch_size,), dtype=torch.int64, device=self._config.device),
            "skill": torch.zeros((batch_size, self._skill_stoch, self._skill_discrete), 
                                device=self._config.device),
            "goal": torch.zeros((batch_size, self._goal_feat_size), device=self._config.device),
        }
    
    def _get_obs_feat(self, state):
        """Extract obs_feat from state dict."""
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            return torch.cat([state["obs_belief"], state["obs_stoch"]], dim=-1)
        else:
            stoch = state["stoch"]
            if self._config.dyn_discrete:
                stoch = stoch.reshape(stoch.shape[:-2] + (-1,))
            return torch.cat([stoch, state["deter"]], dim=-1)
    
    def _get_abs_feat(self, state):
        """Extract abs_feat from state dict."""
        # Support using obs_feat for manager input (debugging/sanity check)
        manager_input = getattr(self._config, 'director_manager_input', 'abs_feat')
        if manager_input == 'obs_feat':
            return self._get_obs_feat(state)
            
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            return torch.cat([state["abs_belief"], state["abs_stoch"]], dim=-1)
        else:
            # For RSSM, use same as obs_feat
            return self._get_obs_feat(state)
    
    def _get_goal_context(self, state):
        """Extract goal context (belief/deter) for conditional goal decoding."""
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            feat = state["obs_belief"]
        else:
            feat = state["deter"]
        # NOTE: Do NOT apply goal_norm here - decoder expects raw features
        return feat
    
    def _get_goal_feat(self, state):
        """
        Extract goal feature for Goal AE (deterministic state only).
        
        CRITICAL: Official Director uses only 'deter' for goal space.
        This excludes stochastic variables which are unstable for goal targeting.
        NOTE: Do NOT normalize - must match Goal AE decoder output space.
        """
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            feat = state["obs_belief"]
        else:
            feat = state["deter"]
        return feat  # Raw features, no normalization
    
    def _should_update_goal(self, state, carry, imag=False):
        """Determine if manager should select a new goal."""
        duration = self._skill_duration
        
        if self._manager_mode == "boundary":
            # VTA boundary-driven: update when boundary is detected
            dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
            if dynamics_type == 'vta' and "boundary" in state:
                is_boundary = state["boundary"].squeeze(-1) > 0.5
                # Also update at start (step == 0)
                is_start = carry["step"] == 0
                return is_boundary | is_start
            else:
                # Fallback to fixed mode
                return (carry["step"] % duration) == 0
        else:
            # Fixed interval mode
            return (carry["step"] % duration) == 0
    
    def policy(self, latent, carry, imag=False):
        """
        Hierarchical policy for environment rollout.
        
        Args:
            latent: Current world model state
            carry: Hierarchical state (step, skill, goal)
            imag: Whether this is imagination rollout
            
        Returns:
            outs: Dict with action distribution
            carry: Updated hierarchical state
        """
        sg = lambda x: {k: v.detach() for k, v in x.items()} if isinstance(x, dict) else x.detach()
        
        # Determine if we should update the goal
        update = self._should_update_goal(latent, carry, imag)
        update_float = update.float()
        
        # Switch function for conditional update
        def switch(old, new):
            # update_float: (batch,) -> expand to match tensor dims
            u = update_float
            while u.dim() < new.dim():
                u = u.unsqueeze(-1)
            return (1 - u) * old + u * new
        
        # Get features
        abs_feat = self._get_abs_feat(latent)
        obs_feat = self._get_obs_feat(latent)
        goal_feat = self._get_goal_feat(latent)  # Deterministic only for goal space
        
        # Get context for conditional goal decoding (obs_belief/deter)
        context = self._get_goal_context(latent)
        
        # Sample skill from manager
        manager_dist = self.manager(sg(abs_feat))
        # MultiCategoricalDist.sample() returns (batch, stoch, discrete) directly
        new_skill = manager_dist.sample()
        skill = switch(carry["skill"], new_skill.detach())
        
        # Decode skill to goal (in goal_feat space = deter only) with context
        new_goal = self.goal_ae.decode(new_skill, context=context.detach()).mode()
        
        if self._use_delta:
            # Goal is delta from current state (using goal_feat, not obs_feat)
            new_goal = goal_feat.detach() + new_goal
        
        goal = switch(carry["goal"], new_goal.detach())
        
        # Worker: obs_feat + goal + abs_feat -> action
        # Worker: obs_feat + goal + [abs_feat] -> action
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        
        input_list = [sg(obs_feat), goal]
        if self._use_delta:
            delta = goal - goal_feat  # Delta in goal space
            input_list.append(delta)
            
        # Only add abs_feat if NOT RSSM (redundant for RSSM as abs_feat=obs_feat)
        if dynamics_type != 'rssm':
            input_list.append(sg(abs_feat))
            
        worker_input = torch.cat(input_list, dim=-1)
        
        action_dist = self.worker(worker_input)
        
        outs = {"action": action_dist}
        
        # Update carry
        new_carry = {
            "step": carry["step"] + 1,
            "skill": skill,
            "goal": goal,
        }
        
        return outs, new_carry
    
    def _goal_reward(self, obs_feat, goal):
        """
        Compute goal-reaching reward for worker.
        
        Args:
            obs_feat: Current observation features
            goal: Target goal features
            
        Returns:
            reward: Goal-reaching reward
        """
        if self._goal_reward_type == "cosine_max" or self._goal_reward_type == "cosine":
            # Normalized cosine similarity with max norm (Official 'cosine' and 'cosine_max')
            gnorm = torch.linalg.norm(goal, dim=-1, keepdim=True) + 1e-12
            fnorm = torch.linalg.norm(obs_feat, dim=-1, keepdim=True) + 1e-12
            norm = torch.maximum(gnorm, fnorm)
            # einsum: sum product over feature dim
            return torch.einsum("...i,...i->...", goal / norm, obs_feat / norm)
        
        elif self._goal_reward_type == "l2":
            # Negative L2 distance
            return -torch.linalg.norm(goal - obs_feat, dim=-1)
        
        elif self._goal_reward_type == "mse":
            # Negative MSE
            return -((goal - obs_feat) ** 2).mean(dim=-1)
        
        else:
            raise NotImplementedError(self._goal_reward_type)
    
    def _expl_reward(self, traj, goal_feat):
        """
        Compute exploration reward.
        
        Official Director (adver_impl='squared'):
            return ((dec.mode() - feat) ** 2).mean(-1)[1:]
        
        Uses Goal AE reconstruction error as exploration bonus.
        High reconstruction error = out-of-distribution state = exploration bonus
        
        Args:
            traj: Trajectory dict (for context extraction)
            goal_feat: Goal features (deter only) to measure novelty
        """
        with torch.no_grad():
            # Official: context = tf.repeat(feat[0][None], 1 + imag_horizon, 0)
            # Use first timestep context for entire trajectory
            context = self._get_goal_context(traj)
            
            # Broadcast first timestep context to match goal_feat length
            T_goal = goal_feat.shape[0]
            context_broadcast = context[0:1].expand(T_goal, *context.shape[1:])
            
            # 1. Encode: goal -> z, q(z|g)
            z, enc_dist = self.goal_ae.encode(goal_feat, context=context_broadcast, sample=True)
            
            # 2. Decode: z -> p(g|z)
            # Returns distribution
            dec_dist = self.goal_ae.decode(z, context=context_broadcast)
            ll = dec_dist.log_prob(goal_feat)
            
            # 3. KL(q(z|g) || p(z))
            # Assume uniform prior for now (as in GoalAutoencoder.loss)
            # KL = sum(p * (log_p - log_prior))
            # Entropy = -sum(p * log_p)
            # KL = -Entropy - sum(p * log_prior) = -Entropy - log_prior (since sum(p)=1)
            # But OneHotDist logic:
            logits = enc_dist.logits
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            # Uniform prior: 1/K
            # log_prior = -log(K)
            kl = (probs * (log_probs + torch.log(torch.tensor(self._skill_discrete, device=probs.device)))).sum(-1).sum(-1) # Sum over stochastic dims
            
            # Official Director Logic Verification
            # Config 'adver_impl' defaults to 'squared' in official repo
            adver_impl = getattr(self._config, 'director_adver_impl', 'squared')
            
            if adver_impl == 'squared':
                 # Official: ((dec - feat) ** 2).mean(-1)
                 # Re-decode to get mode for squared error
                 # This handles "raw" reconstruction error as reward
                 # CRITICAL FIX: Use context_broadcast, not context
                 dec_dist = self.goal_ae.decode(z)
                 rec = dec_dist.mode()
                 return ((rec - goal_feat) ** 2).mean(-1)
                 
            elif adver_impl == 'kl_ll':
                 # Fallback / Previous approach
                 return kl - ll
            
            else:
                 raise NotImplementedError(f"Unknown adver_impl: {adver_impl}")
    
    def _train(self, start, objective):
        """
        Train hierarchical policy in imagination (Official Director train_jointly).
        
        Official structure:
        - Single forward pass through imagination with hierarchical policy
        - split_traj for Worker (K+1 segments)
        - abstract_traj for Manager (coarse segments)
        - Shared gradient flow
        """
        metrics = {}
        
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))\
            if x.dim() > 2 else x
        start = {k: flatten(v) for k, v in start.items()}
        batch_size = list(start.values())[0].shape[0]
        
        # Goal AE training moved to dreamer.py _train() as replay-based training
        # (Official Director: vae_replay=True, vae_imag=False)
        # The goal AE is now trained on temporal sequences from replay buffer
        # where context=state[t] and goal=state[t+K], not on flattened start states
        
        # === TRAIN JOINTLY (Official Director Structure) ===
        with tools.RequiresGrad(self.worker):
            with tools.RequiresGrad(self.manager):
                with torch.cuda.amp.autocast(self._use_amp):
                    # 1. Imagine trajectory with hierarchical policy (carry state)
                    traj, carry_traj = self._imagine_with_carry(
                        start, self._config.imag_horizon
                    )
                    
                    # 2. Compute all rewards
                    goal_feats = self._get_goal_feat(traj)
                    
                    reward_extr = objective(
                        dynamics.get_feat(traj),
                        traj,
                        traj["action"]
                    )
                    
                    # Official: context = tf.repeat(feat[0][None], ...)
                    # Pass original traj for context (uses index 0), but goal_feat[1:] for reward
                    reward_expl = self._expl_reward(traj, goal_feats[1:])
                    
                    # Goal reward: compare current feat to goal in carry
                    reward_goal = self._goal_reward(goal_feats[1:], carry_traj["goal"][1:])
                    
                    # Store in trajectory
                    traj["reward_extr"] = reward_extr
                    traj["reward_expl"] = reward_expl
                    traj["reward_goal"] = reward_goal
                    traj["goal"] = carry_traj["goal"]
                    traj["skill"] = carry_traj["skill"]
                    traj["delta"] = carry_traj["goal"] - goal_feats
                    
                    # Discount/cont handling - add to traj BEFORE split_traj
                    discount = self._config.discount
                    if "cont" not in traj:
                        traj["cont"] = torch.ones_like(reward_extr)
                    # Add discount to traj so it gets properly unfolded in split_traj
                    traj["discount"] = discount * traj["cont"]
                    
                    # 3. Prepare Worker trajectory (split_traj with weight)
                    wtraj, _ = self._split_traj(traj, carry_traj)
                    
                    # 4. Prepare Manager trajectory (abstract_traj)
                    mtraj, mcarry = self._abstract_traj_official(traj, carry_traj)
                    
                    # 5. Update Worker (using wtraj['weight'])
                    w_obs = self._get_obs_feat(wtraj)
                    w_abs = self._get_abs_feat(wtraj)
                    w_goal = wtraj["goal"]
                    w_delta = wtraj.get("delta", wtraj["goal"] - self._get_goal_feat(wtraj))
                    
                    dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
                    
                    w_input_list = [w_obs, w_goal]
                    if self._use_delta:
                        w_input_list.append(w_delta)
                        
                    # Only add abs_feat if NOT RSSM (redundant for RSSM)
                    if dynamics_type != 'rssm':
                        w_input_list.append(w_abs)
                        
                    w_input = torch.cat(w_input_list, dim=-1)
                    
                    worker_loss, worker_mets = self.worker.update(
                        w_input, wtraj, wtraj['weight']
                    )
                    metrics.update({f'worker_{k}': v for k, v in worker_mets.items()})
                    
                    # 6. Update Manager (using mtraj['weight'])
                    if "reward_extr" in mtraj and mtraj["reward_extr"].shape[0] > 0:
                        m_abs = self._get_abs_feat(mtraj)
                        manager_loss, manager_mets = self.manager.update(
                            m_abs, mtraj, mtraj.get('weight', torch.ones_like(mtraj['reward_extr']))
                        )
                        metrics.update({f'manager_{k}': v for k, v in manager_mets.items()})
                    else:
                        manager_loss = torch.tensor(0.0, device=self._config.device)
                # Optimize Worker (first backward)
                metrics.update(self._worker_opt(worker_loss, self.worker.parameters()))
                
                # Optimize Manager (second backward)
                # Manager uses different trajectory (mtraj from abstract_traj), so gradients should be independent
                if "reward_extr" in mtraj and mtraj["reward_extr"].shape[0] > 0:
                    metrics.update(self._manager_opt(manager_loss, self.manager.parameters()))
        
        self._update_count += 1
        
        # Debug metrics
        metrics["worker_goal_reward_mean"] = reward_goal.mean().item() if reward_goal.numel() > 0 else 0.0
        metrics["manager_reward_mean"] = reward_extr.mean().item() if reward_extr.numel() > 0 else 0.0
        
        return None, None, None, None, metrics





    def _imagine_with_carry(self, start, horizon):
        """
        Imagine trajectory with hierarchical policy carry state.
        
        Args:
            start: Start state dict from world model posterior
            horizon: Number of imagination steps
            
        Returns:
            traj: Trajectory dict with states and actions
            carry_traj: Carry state trajectory with skills and goals
        """
        dynamics = self._world_model.dynamics
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        batch_size = list(start.values())[0].shape[0]
        
        # Initialize
        carry = self.initial(batch_size)
        state = start
        
        states = {k: [v] for k, v in start.items()}
        carries = {k: [v] for k, v in carry.items()}
        actions = []
        
        for t in range(horizon):
            # Get action from hierarchical policy
            outs, carry = self.policy(state, carry, imag=True)
            action = outs["action"].sample()
            actions.append(action)
            
            # World model imagination step
            if dynamics_type == 'vta':
                boundary_mode = getattr(self._config, 'vta_imag_boundary', 'prior')
                state = dynamics.img_step(state, action, boundary_mode=boundary_mode)
            else:
                state = dynamics.img_step(state, action)
            
            for k, v in state.items():
                states[k].append(v)
            for k, v in carry.items():
                carries[k].append(v)
        
        # Stack trajectories
        traj = {k: torch.stack(v, dim=0) for k, v in states.items()}
        traj["action"] = torch.stack(actions, dim=0)
        carry_traj = {k: torch.stack(v, dim=0) for k, v in carries.items()}
        
        return traj, carry_traj
    
    def _split_traj(self, traj, carry_traj):
        """
        Split trajectory for Worker training (OFFICIAL Director implementation).
        
        Official algorithm:
        1. Reshape trajectory into (N, K, B, ...) segments where N = H // K
        2. Create overlapping K+1 segments by appending boundary states
        3. Transpose to (K+1, N*B, ...)
        
        Args:
            traj: Trajectory dict (states, rewards, actions)
            carry_traj: Carry state trajectory (skill, goal) - merged into traj
            
        Returns:
            wtraj: Worker trajectory (K+1 length segments, flattened batch)
            wcarry: Empty dict (not used in official)
        """
        K = self._skill_duration
        traj = {**traj}  # Copy to avoid mutation
        
        # Merge carry_traj into traj (official does this before split)
        for key, val in carry_traj.items():
            if val is not None and key not in traj:
                traj[key] = val
        
        # Get dimensions
        # States have H+1 length, actions/rewards have H length
        H = traj['action'].shape[0] if 'action' in traj else (list(traj.values())[0].shape[0] - 1)
        B = list(traj.values())[0].shape[1]
        
        # Official: assert len(traj['action']) % k == 1  (for states: H+1)
        # This means H must be divisible by K
        # If not, truncate to nearest multiple
        N = H // K  # Number of complete segments
        if N < 1:
            # Can't split, return traj as-is
            traj['weight'] = torch.cumprod(
                self._config.discount * traj.get('cont', torch.ones_like(traj['reward_goal'])),
                dim=0
            ) / self._config.discount
            return traj, {}
        
        usable_H = N * K
        
        wtraj = {}
        
        # Official reshape function
        def reshape_fn(x, is_state):
            """
            Reshape to (N, K, B, ...) for non-state or (N, K+1, B, ...) for state
            """
            if is_state:
                # States: length H+1, take first N*K+1 elements
                return x[:usable_H + 1]
            else:
                # Actions/Rewards: length H, take first N*K elements
                return x[:usable_H]
        
        for key, val in traj.items():
            if val is None:
                continue
                
            T = val.shape[0]
            is_reward = 'reward' in key
            # is_state only if NOT reward and length matches H+1
            is_state = (T == H + 1) and not is_reward
            
            # Truncate to usable length
            if is_state:
                val_use = val[:usable_H + 1]  # N*K + 1
            else:
                val_use = val[:usable_H]  # N*K
            
            # Official: Pad rewards with 0 at start
            # val = tf.concat([0 * val[:1], val], 0) if 'reward' in key else val
            if is_reward:
                val_use = torch.cat([torch.zeros_like(val_use[:1]), val_use], dim=0)
            
            # Official reshape logic:
            # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
            # val = tf.concat([reshape(val[:-1]), val[k::k][:, None]], 1)
            
            if is_reward:
                # After padding, rewards have N*K + 1 length
                # Reshape: (N*K, B, ...) -> (N, K, B, ...)
                reshaped = val_use[:-1].reshape(N, K, B, *val.shape[2:])
                # Append boundary: val[k::k] = elements at K, 2K, 3K, ... (N elements)
                boundary = val_use[K::K]  # (N, B, ...)
                # Concatenate: (N, K, B, ...) + (N, 1, B, ...) -> (N, K+1, B, ...)
                combined = torch.cat([reshaped, boundary.unsqueeze(1)], dim=1)
            elif is_state:
                # States have N*K + 1 length
                # Reshape first N*K elements: (N*K, B, ...) -> (N, K, B, ...)
                reshaped = val_use[:-1].reshape(N, K, B, *val.shape[2:])
                # Boundary: val[k::k] = states at K, 2K, ... (includes final state)
                boundary = val_use[K::K]  # Should be (N, B, ...) or (N+1, B, ...)
                if boundary.shape[0] > N:
                    boundary = boundary[:N]
                combined = torch.cat([reshaped, boundary.unsqueeze(1)], dim=1)
            else:
                # Actions: N*K length
                # Reshape: (N*K, B, ...) -> (N, K, B, ...)
                reshaped = val_use.reshape(N, K, B, *val.shape[2:])
                # Boundary: val[k::k] = every Kth action starting from K
                # But we only have N*K actions, so val[K::K] has N-1 elements
                # Need to handle this carefully
                boundary_indices = list(range(K, usable_H, K))  # [K, 2K, ..., (N-1)*K]
                if len(boundary_indices) < N:
                    # Use last action for final boundary
                    boundary = torch.cat([val_use[boundary_indices], val_use[-1:]], dim=0)
                else:
                    boundary = val_use[boundary_indices[:N]]
                combined = torch.cat([reshaped, boundary.unsqueeze(1)], dim=1)
            
            # Official: Transpose to (K+1, N, B, ...) then reshape to (K+1, N*B, ...)
            # val = val.transpose([1, 0] + list(range(2, len(val.shape))))
            # val = val.reshape([val.shape[0], np.prod(val.shape[1:3])] + val.shape[3:])
            
            # combined: (N, K+1, B, ...)
            # Transpose: (K+1, N, B, ...)
            perm = [1, 0] + list(range(2, combined.dim()))
            transposed = combined.permute(perm)
            
            # Reshape: (K+1, N*B, ...)
            final = transposed.reshape(transposed.shape[0], N * B, *transposed.shape[3:])
            
            # Official: Remove padding from rewards
            # val = val[1:] if 'reward' in key else val
            if is_reward:
                final = final[1:]
            
            wtraj[key] = final
        
        # Official: Bootstrap sub trajectory against current not next goal
        # traj['goal'] = tf.concat([traj['goal'][:-1], traj['goal'][:1]], 0)
        if 'goal' in wtraj and wtraj['goal'].shape[0] > 1:
            goal = wtraj['goal']
            wtraj['goal'] = torch.cat([goal[:-1], goal[:1]], dim=0)
        
        # Official weight computation:
        # traj['weight'] = tf.math.cumprod(self.config.discount * traj['cont']) / self.config.discount
        discount = self._config.discount
        if 'cont' in wtraj:
            wtraj['weight'] = torch.cumprod(discount * wtraj['cont'], dim=0) / discount
        else:
            # Create default cont of 1s
            ref = next(iter(wtraj.values()))
            cont = torch.ones(ref.shape[0], ref.shape[1], device=ref.device)
            wtraj['weight'] = torch.cumprod(discount * cont, dim=0) / discount
        
        return wtraj, {}
    

    
    def _abstract_traj_official(self, traj, carry_traj):
        """
        Abstract trajectory for Manager training (official Director implementation).
        
        Aggregates rewards over skill duration and samples states at segment boundaries.
        Manager sees a coarser time resolution than Worker.
        
        Official implementation:
        - traj['action'] = traj.pop('skill')
        - rewards: weighted mean over each segment using cumprod of cont
        - cont: product over each segment
        - other keys: sample at segment boundaries (starts + final)
        - weight: cumprod(discount * cont) / discount (same discount as Worker)
        """
        K = self._skill_duration
        H = traj["action"].shape[0] if "action" in traj else (list(traj.values())[0].shape[0] - 1)
        B = list(traj.values())[0].shape[1]
        
        # Number of complete segments
        N = H // K
        if N < 1:
            return traj, carry_traj
        
        usable_H = N * K
        
        # Copy traj and merge carry
        mtraj = {}
        
        # Official: traj['action'] = traj.pop('skill')
        # We need to get skill from carry_traj
        if 'skill' in carry_traj:
            # Skill at segment starts: indices 0, K, 2K, ...
            skill = carry_traj['skill']
            skill_indices = list(range(0, usable_H, K))
            mtraj['action'] = skill[skill_indices]  # (N, B, stoch, discrete)
        
        # Official: Compute weights for reward aggregation
        # weights = tf.math.cumprod(reshape(traj['cont'][:-1]), 1)
        # Get cont from traj
        if 'cont' in traj:
            cont = traj['cont']
        else:
            cont = torch.ones((H + 1, B), device=list(traj.values())[0].device)
        
        # Reshape cont[:-1] (first H elements) to (N, K, B)
        if cont.dim() > 2:
            cont = cont.squeeze(-1)
        cont_usable = cont[:usable_H]  # (N*K, B)
        cont_seg = cont_usable.reshape(N, K, B)
        # Compute cumprod within each segment for weighting
        weights_seg = torch.cumprod(cont_seg, dim=1)  # (N, K, B)
        
        # Official reshape function
        def reshape_fn(x):
            return x[:usable_H].reshape(N, K, B, *x.shape[2:])
        
        # Process each key
        for key, val in traj.items():
            if val is None:
                continue
            
            # Skip 'action' - Manager's action is skill, set separately
            if key == 'action':
                continue
            
            T = val.shape[0]
            is_state = (T == H + 1)
            is_reward = 'reward' in key
            
            if is_reward:
                # Official: traj[key] = (reshape(value) * weights).mean(1)
                # Weighted mean over segment
                val_usable = val[:usable_H]  # (N*K, B, ...)
                val_seg = val_usable.reshape(N, K, B, *val.shape[2:])  # (N, K, B, ...)
                
                # weights_seg is (N, K, B), need to broadcast to val_seg shape
                if val_seg.dim() > 3:
                    # Add dimensions for feature dims
                    w = weights_seg
                    for _ in range(val_seg.dim() - 3):
                        w = w.unsqueeze(-1)
                else:
                    w = weights_seg
                
                # Weighted mean: (val * weights).mean(dim=1)
                result = (val_seg * w).mean(dim=1)  # (N, B, ...)
                # Squeeze trailing dimension if present (e.g., (N, B, 1) -> (N, B))
                if result.dim() > 2 and result.shape[-1] == 1:
                    result = result.squeeze(-1)
                mtraj[key] = result
                
            elif key == 'cont':
                # Official: traj[key] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)
                # Squeeze trailing dimension if present (e.g., (T, B, 1) -> (T, B))
                val_squeezed = val.squeeze(-1) if val.dim() > 2 else val
                cont_first = val_squeezed[:1]  # (1, B)
                cont_rest = val_squeezed[1:usable_H + 1]  # (N*K, B)
                cont_rest_seg = cont_rest.reshape(N, K, B)
                cont_prod = cont_rest_seg.prod(dim=1)  # (N, B)
                mtraj[key] = torch.cat([cont_first, cont_prod], dim=0)  # (N+1, B)
                
            elif is_state:
                # Official: traj[key] = tf.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
                # Take first element of each segment + final state
                val_usable = val[:usable_H]  # (N*K, B, ...)
                val_seg = val_usable.reshape(N, K, B, *val.shape[2:])
                val_starts = val_seg[:, 0]  # (N, B, ...)
                val_final = val[usable_H:usable_H + 1]  # (1, B, ...)
                mtraj[key] = torch.cat([val_starts, val_final], dim=0)  # (N+1, B, ...)
            
            else:
                # Non-state, non-reward (e.g., original action - but we override with skill)
                # Official: sample at segment starts
                val_usable = val[:usable_H]
                val_seg = val_usable.reshape(N, K, B, *val.shape[2:])
                mtraj[key] = val_seg[:, 0]  # (N, B, ...)
        
        # Process carry_traj for goal
        mcarry = {}
        if 'goal' in carry_traj:
            goal = carry_traj['goal']
            # Sample at segment boundaries: 0, K, 2K, ..., N*K
            goal_indices = list(range(0, usable_H + 1, K))
            mcarry['goal'] = goal[goal_indices]  # (N+1, B, ...)
            mtraj['goal'] = mcarry['goal']  # Also add to mtraj for compatibility
        
        if 'skill' in carry_traj:
            skill = carry_traj['skill']
            skill_indices = list(range(0, usable_H + 1, K))
            mcarry['skill'] = skill[skill_indices]
        
        # Official weight computation (same discount as Worker, NOT gamma^K):
        # traj['weight'] = tf.math.cumprod(self.config.discount * traj['cont']) / self.config.discount
        discount = self._config.discount  # NOT gamma^K
        if 'cont' in mtraj:
            mtraj['weight'] = torch.cumprod(discount * mtraj['cont'], dim=0) / discount
        else:
            ref = next(iter(mtraj.values()))
            T = ref.shape[0]
            cont_default = torch.ones((T, B), device=ref.device)
            mtraj['weight'] = torch.cumprod(discount * cont_default, dim=0) / discount
        
        return mtraj, mcarry
    
    # _update_slow_target is handled by ImagActorCritic internally

