"""
Variational Temporal Abstraction (VTA) for DreamerV3

Based on "Variational Temporal Abstraction" by Kim, Ahn, Bengio (NeurIPS 2019)
https://arxiv.org/abs/1910.00775

This module implements a hierarchical recurrent state space model that:
1. Detects temporal boundaries (segment switches) in sequences
2. Maintains abstract states (z) that update only at boundaries
3. Maintains observation states (s) that update every timestep
4. Enables "jumpy imagination" for efficient agent learning
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from . import tools


def gumbel_sampling(log_alpha, temp=1.0, eps=1e-8):
    """Sample from Gumbel-Softmax distribution."""
    u = torch.rand_like(log_alpha).clamp(eps, 1 - eps)
    gumbel = -torch.log(-torch.log(u))
    return (log_alpha + gumbel) / temp


class PriorBoundaryDetector(nn.Module):
    """Predicts boundary probability from observation features (prior)."""
    
    def __init__(self, input_size, hidden_size, act="SiLU", norm=True):
        super().__init__()
        act_fn = getattr(torch.nn, act)
        layers = []
        layers.append(nn.Linear(input_size, hidden_size, bias=False))
        if norm:
            layers.append(nn.LayerNorm(hidden_size, eps=1e-03))
        layers.append(act_fn())
        layers.append(nn.Linear(hidden_size, 2))  # 2 classes: READ (boundary), COPY (no boundary)
        self.network = nn.Sequential(*layers)
        self.network.apply(tools.weight_init)
    
    def forward(self, obs_feat):
        """
        Args:
            obs_feat: (batch, feat_size) observation features
        Returns:
            log_alpha: (batch, 2) log probabilities for [READ, COPY]
        """
        return self.network(obs_feat)


class PostBoundaryDetector(nn.Module):
    """Infers boundary from full encoded sequence (posterior)."""
    
    def __init__(self, input_size, hidden_size, num_layers=2, act="SiLU", norm=True):
        super().__init__()
        act_fn = getattr(torch.nn, act)
        
        # 1D convolution over time dimension
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(nn.Conv1d(in_ch, hidden_size, kernel_size=3, padding=1, bias=not norm))
            if norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn())
        layers.append(nn.Conv1d(hidden_size, 2, kernel_size=3, padding=1))
        self.network = nn.Sequential(*layers)
        self.network.apply(tools.weight_init)
    
    def forward(self, enc_obs_list):
        """
        Args:
            enc_obs_list: (batch, time, feat_size) encoded observations
        Returns:
            log_alpha: (batch, time, 2) log probabilities for [READ, COPY]
        """
        # (batch, time, feat) -> (batch, feat, time)
        x = enc_obs_list.permute(0, 2, 1)
        x = self.network(x)
        # (batch, 2, time) -> (batch, time, 2)
        return x.permute(0, 2, 1)


class LatentDistribution(nn.Module):
    """Gaussian distribution layer for latent states."""
    
    def __init__(self, input_size, latent_size, hidden_size=None, act="SiLU", norm=True, min_std=0.1):
        super().__init__()
        self._min_std = min_std
        self._latent_size = latent_size
        
        if hidden_size is None:
            hidden_size = input_size
        
        act_fn = getattr(torch.nn, act)
        
        # Feature extraction
        if hidden_size == input_size:
            self.feat = nn.Identity()
        else:
            layers = []
            layers.append(nn.Linear(input_size, hidden_size, bias=False))
            if norm:
                layers.append(nn.LayerNorm(hidden_size, eps=1e-03))
            layers.append(act_fn())
            self.feat = nn.Sequential(*layers)
        
        # Mean and std layers
        self.mean_layer = nn.Linear(hidden_size, latent_size)
        self.std_layer = nn.Linear(hidden_size, latent_size)
        
        self.mean_layer.apply(tools.uniform_weight_init(1.0))
        self.std_layer.apply(tools.uniform_weight_init(1.0))
    
    def forward(self, input_data):
        """
        Returns Normal distribution with computed mean and std.
        """
        feat = self.feat(input_data)
        mean = self.mean_layer(feat)
        std = F.softplus(self.std_layer(feat)) + self._min_std
        return {"mean": mean, "std": std}
    
    def get_dist(self, stats):
        mean = stats["mean"]
        std = stats["std"]
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(std, nan=self._min_std, posinf=1.0, neginf=self._min_std)
        std = torch.clamp(std, min=self._min_std, max=1e2)
        return torchd.normal.Normal(mean, std)
    
    def sample(self, stats):
        dist = self.get_dist(stats)
        return dist.rsample()


class VTA(nn.Module):
    """
    Variational Temporal Abstraction (VTA) - Hierarchical State Space Model
    
    This replaces RSSM in DreamerV3 with a two-level hierarchical model:
    - Abstract level: z_t (updates only at temporal boundaries)
    - Observation level: s_t (updates every timestep)
    """
    
    def __init__(
        self,
        abs_belief=512,
        abs_stoch=32,
        obs_belief=512,
        obs_stoch=32,
        hidden=512,
        num_layers=2,
        max_seg_len=10,
        max_seg_num=5,
        boundary_temp=1.0,
        boundary_force_scale=10.0,
        act="SiLU",
        norm=True,
        min_std=0.1,
        num_actions=None,
        embed_size=None,
        device=None,
        vta_posterior_input='embed',
    ):
        super().__init__()
        
        # Dimensions
        self._abs_belief_size = abs_belief
        self._abs_stoch_size = abs_stoch
        self._obs_belief_size = obs_belief
        self._obs_stoch_size = obs_stoch
        self._hidden = hidden
        self._num_layers = num_layers
        self._num_actions = num_actions
        self._embed_size = embed_size
        self._device = device
        self._min_std = min_std
        
        # Segment constraints
        self._max_seg_len = max_seg_len
        self._max_seg_num = max_seg_num
        self._boundary_temp = boundary_temp
        self._boundary_force_scale = boundary_force_scale
        
        # Feature sizes
        self._abs_feat_size = abs_belief + abs_stoch
        self._obs_feat_size = obs_belief + obs_stoch
        
        act_fn = getattr(torch.nn, act)
        
        # ========================
        # Boundary Detectors
        # ========================
        # ========================
        # Boundary Detectors
        # ========================
        self.prior_boundary = PriorBoundaryDetector(
            input_size=self._obs_feat_size,
            hidden_size=hidden,
            act=act,
            norm=norm,
        )
        self._posterior_input = vta_posterior_input if vta_posterior_input else 'embed'
        post_input_size = embed_size
        if self._posterior_input == 'embed_reward':
            post_input_size += 1

        self.post_boundary = PostBoundaryDetector(
            input_size=post_input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            act=act,
            norm=norm,
        )
        
        # ========================
        # Abstract Level
        # ========================
        # Input layer: prev_abs_stoch + action -> hidden
        abs_inp_dim = abs_stoch + num_actions
        abs_inp_layers = []
        abs_inp_layers.append(nn.Linear(abs_inp_dim, hidden, bias=False))
        if norm:
            abs_inp_layers.append(nn.LayerNorm(hidden, eps=1e-03))
        abs_inp_layers.append(act_fn())
        self._abs_inp_layers = nn.Sequential(*abs_inp_layers)
        self._abs_inp_layers.apply(tools.weight_init)
        
        # Abstract belief RNN (GRU)
        self._abs_cell = GRUCell(hidden, abs_belief, norm=norm)
        self._abs_cell.apply(tools.weight_init)
        
        # Prior over abstract state: p(z_t | abs_belief_t)
        self.prior_abs_state = LatentDistribution(
            input_size=abs_belief,
            latent_size=abs_stoch,
            hidden_size=hidden,
            act=act,
            norm=norm,
            min_std=min_std,
        )
        
        # Posterior over abstract state: q(z_t | abs_belief_t, embed)
        self.post_abs_state = LatentDistribution(
            input_size=abs_belief + embed_size,
            latent_size=abs_stoch,
            hidden_size=hidden,
            act=act,
            norm=norm,
            min_std=min_std,
        )
        
        # Initial abstract belief (learned)
        self.init_abs_belief = nn.Parameter(
            torch.zeros((1, abs_belief), device=device),
            requires_grad=True,
        )
        
        # ========================
        # Observation Level
        # ========================
        # Input layer: prev_obs_stoch + abs_feat -> hidden
        obs_inp_dim = obs_stoch + self._abs_feat_size
        obs_inp_layers = []
        obs_inp_layers.append(nn.Linear(obs_inp_dim, hidden, bias=False))
        if norm:
            obs_inp_layers.append(nn.LayerNorm(hidden, eps=1e-03))
        obs_inp_layers.append(act_fn())
        self._obs_inp_layers = nn.Sequential(*obs_inp_layers)
        self._obs_inp_layers.apply(tools.weight_init)
        
        # Observation belief RNN (GRU)
        self._obs_cell = GRUCell(hidden, obs_belief, norm=norm)
        self._obs_cell.apply(tools.weight_init)
        
        # Prior over observation state: p(s_t | obs_belief_t)
        self.prior_obs_state = LatentDistribution(
            input_size=obs_belief,
            latent_size=obs_stoch,
            hidden_size=hidden,
            act=act,
            norm=norm,
            min_std=min_std,
        )
        
        # Posterior over observation state: q(s_t | obs_belief_t, abs_feat, embed)
        self.post_obs_state = LatentDistribution(
            input_size=obs_belief + self._abs_feat_size + embed_size,
            latent_size=obs_stoch,
            hidden_size=hidden,
            act=act,
            norm=norm,
            min_std=min_std,
        )
        
        # Initial observation belief from abstract features
        init_obs_layers = []
        init_obs_layers.append(nn.Linear(self._abs_feat_size, obs_belief, bias=False))
        if norm:
            init_obs_layers.append(nn.LayerNorm(obs_belief, eps=1e-03))
        init_obs_layers.append(act_fn())
        self._init_obs_belief = nn.Sequential(*init_obs_layers)
        self._init_obs_belief.apply(tools.weight_init)
    
    def initial(self, batch_size):
        """Initialize hierarchical state."""
        device = self._device
        
        # Abstract level
        abs_belief = self.init_abs_belief.repeat(batch_size, 1)
        abs_stoch = torch.zeros(batch_size, self._abs_stoch_size, device=device)
        abs_mean = torch.zeros(batch_size, self._abs_stoch_size, device=device)
        abs_std = torch.ones(batch_size, self._abs_stoch_size, device=device) * self._min_std
        
        # Observation level
        obs_belief = torch.zeros(batch_size, self._obs_belief_size, device=device)
        obs_stoch = torch.zeros(batch_size, self._obs_stoch_size, device=device)
        obs_mean = torch.zeros(batch_size, self._obs_stoch_size, device=device)
        obs_std = torch.ones(batch_size, self._obs_stoch_size, device=device) * self._min_std
        
        # Boundary info
        boundary = torch.ones(batch_size, 1, device=device)  # Start with boundary
        boundary_logit = torch.zeros(batch_size, 2, device=device)
        
        # Segment tracking
        seg_len = torch.zeros(batch_size, 1, device=device)
        seg_num = torch.zeros(batch_size, 1, device=device)
        
        return {
            "abs_belief": abs_belief,
            "abs_stoch": abs_stoch,
            "abs_mean": abs_mean,
            "abs_std": abs_std,
            "obs_belief": obs_belief,
            "obs_stoch": obs_stoch,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "boundary": boundary,
            "boundary_logit": boundary_logit,
            "seg_len": seg_len,
            "seg_num": seg_num,
        }
    
    def _get_abs_feat(self, state):
        """Get abstract features from state."""
        return torch.cat([state["abs_belief"], state["abs_stoch"]], dim=-1)
    
    def _get_obs_feat(self, state):
        """Get observation features from state."""
        return torch.cat([state["obs_belief"], state["obs_stoch"]], dim=-1)
    
    def get_feat(self, state):
        """
        Get combined features for decoder/heads.
        Returns concatenation of abstract and observation features.
        """
        abs_feat = self._get_abs_feat(state)
        obs_feat = self._get_obs_feat(state)
        return torch.cat([abs_feat, obs_feat], dim=-1)
    
    @property
    def feat_size(self):
        """Total feature size for decoder/heads."""
        return self._abs_feat_size + self._obs_feat_size
    
    def get_dist(self, state, level="obs"):
        """Get distribution for KL computation."""
        if level == "abs":
            mean, std = state["abs_mean"], state["abs_std"]
        else:
            mean, std = state["obs_mean"], state["obs_std"]
        return tools.ContDist(
            torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
        )
    
    def _sample_boundary(self, log_alpha, temp=None):
        """Sample boundary using Gumbel-Softmax."""
        if temp is None:
            temp = self._boundary_temp
        
        if self.training:
            log_sample = gumbel_sampling(log_alpha, temp=temp)
        else:
            log_sample = log_alpha / temp
        
        # Normalize
        log_sample = log_sample - torch.logsumexp(log_sample, dim=-1, keepdim=True)
        sample_prob = log_sample.exp()
        
        # Hard sample with straight-through estimator
        hard_sample = torch.zeros_like(sample_prob)
        hard_sample.scatter_(-1, sample_prob.argmax(dim=-1, keepdim=True), 1.0)
        sample = hard_sample.detach() + (sample_prob - sample_prob.detach())
        
        return sample, log_sample
    
    def _regularize_boundary(self, log_alpha, seg_len, seg_num):
        """Regularize boundary logits based on segment constraints."""
        max_scale = float(self._boundary_force_scale)
        if max_scale > 0.0:
            # Force READ if segment too long
            over_len = (seg_len >= self._max_seg_len).float()
            force_read = torch.stack(
                [
                    torch.ones_like(seg_len) * max_scale,
                    torch.ones_like(seg_len) * -max_scale,
                ],
                dim=-1,
            ).squeeze(-2)

            # Force COPY if too many segments
            over_num = (seg_num >= self._max_seg_num).float()
            force_copy = torch.stack(
                [
                    torch.ones_like(seg_len) * -max_scale,
                    torch.ones_like(seg_len) * max_scale,
                ],
                dim=-1,
            ).squeeze(-2)

            # Apply constraints
            log_alpha = over_len * force_read + (1 - over_len) * log_alpha
            log_alpha = over_num * force_copy + (1 - over_num) * log_alpha
        
        return log_alpha
    
    def observe(self, embed, action, is_first, state=None, reward=None):
        """
        Process observation sequence with boundary detection (training).
        
        Args:
            embed: (batch, time, embed_size) encoded observations
            action: (batch, time, action_size) actions
            is_first: (batch, time, 1) episode start flags
            state: optional initial state
            reward: (batch, time) optional reward sequence
            
        Returns:
            post: posterior states dict
            prior: prior states dict
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        
        batch_size, seq_len = embed.shape[:2]
        
        # Swap to (time, batch, ...)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        
        # Get posterior boundary predictions for full sequence
        # Prepare input for posterior boundary
        if self._posterior_input == 'embed_reward' and reward is not None:
             # reward: (batch, time) -> (batch, time, 1) -> swap -> (time, batch, 1)
            reward_inp = swap(reward).unsqueeze(-1)
            post_inp = torch.cat([embed, reward_inp], dim=-1)
        else:
            post_inp = embed

        post_boundary_logits = self.post_boundary(swap(post_inp))  # (batch, time, 2)
        post_boundary_logits = swap(post_boundary_logits)  # (time, batch, 2)
        
        # Initialize state if needed
        if state is None:
            state = self.initial(batch_size)
        
        # Collect states
        posts = {k: [] for k in state.keys()}
        priors = {k: [] for k in state.keys()}
        
        for t in range(seq_len):
            post, prior = self.obs_step(
                state, action[t], embed[t], is_first[t],
                post_boundary_logit=post_boundary_logits[t]
            )
            
            for k in state.keys():
                posts[k].append(post[k])
                priors[k].append(prior[k])
            
            state = post
        
        # Stack and swap back to (batch, time, ...)
        posts = {k: swap(torch.stack(v, dim=0)) for k, v in posts.items()}
        priors = {k: swap(torch.stack(v, dim=0)) for k, v in priors.items()}
        
        return posts, priors
    
    def obs_step(self, prev_state, prev_action, embed, is_first, post_boundary_logit=None, sample=True):
        """
        Single observation step with boundary detection.
        
        Args:
            prev_state: previous hierarchical state
            prev_action: previous action
            embed: current observation embedding
            is_first: whether this is the first step
            post_boundary_logit: posterior boundary logit (optional, for training)
            sample: whether to sample or use mode
            
        Returns:
            post: posterior state
            prior: prior state
        """
        batch_size = embed.shape[0]
        
        # Reset state if is_first
        if prev_state is None or torch.all(is_first):
            prev_state = self.initial(batch_size)
            prev_action = torch.zeros(batch_size, self._num_actions, device=self._device)
        elif torch.any(is_first):
            init_state = self.initial(batch_size)
            # Ensure is_first is broadcastable: (batch, 1) or (batch,)
            is_first_flat = is_first.view(batch_size, -1)  # (batch, 1)
            for key in prev_state.keys():
                # Broadcast is_first to match each tensor's shape
                is_first_expanded = is_first_flat
                while is_first_expanded.ndim < prev_state[key].ndim:
                    is_first_expanded = is_first_expanded.unsqueeze(-1)
                # Expand to same size as the target tensor
                is_first_expanded = is_first_expanded.expand_as(prev_state[key])
                prev_state[key] = prev_state[key] * (1.0 - is_first_expanded) + \
                                  init_state[key] * is_first_expanded
            # Broadcast for action: (batch, 1) -> (batch, action_dim)
            is_first_action = is_first_flat.expand_as(prev_action)
            prev_action = prev_action * (1.0 - is_first_action)
        
        # Get prior boundary decision (always compute for kl_mask)
        obs_feat = self._get_obs_feat(prev_state)
        prior_boundary_logit = self.prior_boundary(obs_feat)
        
        # Get boundary decision for sampling
        if post_boundary_logit is not None:
            # Use posterior boundary (training)
            boundary_logit = post_boundary_logit
        else:
            # Use prior boundary (inference)
            boundary_logit = prior_boundary_logit
        
        # Regularize and sample boundary
        boundary_logit = self._regularize_boundary(
            boundary_logit, prev_state["seg_len"], prev_state["seg_num"]
        )
        boundary_sample, boundary_log = self._sample_boundary(boundary_logit)
        
        # boundary_sample[:, 0] = 1 means READ (boundary), [:, 1] = 1 means COPY
        read_mask = boundary_sample[:, 0:1]  # (batch, 1)
        copy_mask = boundary_sample[:, 1:2]  # (batch, 1)
        
        # Update segment tracking
        seg_len = read_mask * 1.0 + copy_mask * (prev_state["seg_len"] + 1.0)
        seg_num = read_mask * (prev_state["seg_num"] + 1.0) + copy_mask * prev_state["seg_num"]
        
        # ========================
        # Abstract level transition
        # ========================
        # Prior: img_step for abstract level
        abs_inp = self._abs_inp_layers(torch.cat([prev_state["abs_stoch"], prev_action], dim=-1))
        abs_belief_updated, _ = self._abs_cell(abs_inp, [prev_state["abs_belief"]])
        abs_belief = read_mask * abs_belief_updated + copy_mask * prev_state["abs_belief"]
        
        prior_abs_stats = self.prior_abs_state(abs_belief)
        prior_abs_dist = self.prior_abs_state.get_dist(prior_abs_stats)
        
        # Posterior abstract state
        post_abs_input = torch.cat([abs_belief, embed], dim=-1)
        post_abs_stats = self.post_abs_state(post_abs_input)
        post_abs_dist = self.post_abs_state.get_dist(post_abs_stats)
        
        if sample:
            post_abs_stoch = post_abs_dist.rsample()
            prior_abs_stoch = prior_abs_dist.rsample()
        else:
            post_abs_stoch = post_abs_dist.mean
            prior_abs_stoch = prior_abs_dist.mean
        
        # Apply COPY mask to abstract state
        abs_stoch = read_mask * post_abs_stoch + copy_mask * prev_state["abs_stoch"]
        
        # Abstract features
        abs_feat = torch.cat([abs_belief, abs_stoch], dim=-1)
        
        # ========================
        # Observation level transition
        # ========================
        # Observation belief: reset on boundary, update otherwise
        obs_inp = self._obs_inp_layers(torch.cat([prev_state["obs_stoch"], abs_feat], dim=-1))
        obs_belief_updated, _ = self._obs_cell(obs_inp, [prev_state["obs_belief"]])
        obs_belief_init = self._init_obs_belief(abs_feat)
        obs_belief = read_mask * obs_belief_init + copy_mask * obs_belief_updated
        
        # Prior observation state
        prior_obs_stats = self.prior_obs_state(obs_belief)
        prior_obs_dist = self.prior_obs_state.get_dist(prior_obs_stats)
        
        # Posterior observation state
        post_obs_input = torch.cat([obs_belief, abs_feat, embed], dim=-1)
        post_obs_stats = self.post_obs_state(post_obs_input)
        post_obs_dist = self.post_obs_state.get_dist(post_obs_stats)
        
        if sample:
            obs_stoch = post_obs_dist.rsample()
            prior_obs_stoch = prior_obs_dist.rsample()
        else:
            obs_stoch = post_obs_dist.mean
            prior_obs_stoch = prior_obs_dist.mean
        
        # Build output states
        # post uses post_boundary_logit, prior uses prior_boundary_logit
        post = {
            "abs_belief": abs_belief,
            "abs_stoch": abs_stoch,
            "abs_mean": post_abs_stats["mean"],
            "abs_std": post_abs_stats["std"],
            "obs_belief": obs_belief,
            "obs_stoch": obs_stoch,
            "obs_mean": post_obs_stats["mean"],
            "obs_std": post_obs_stats["std"],
            "boundary": read_mask,
            "boundary_logit": boundary_logit,  # post boundary logit (regularized)
            "seg_len": seg_len,
            "seg_num": seg_num,
        }
        
        # Regularize prior boundary logit for storing
        prior_boundary_logit_reg = self._regularize_boundary(
            prior_boundary_logit, prev_state["seg_len"], prev_state["seg_num"]
        )
        
        prior = {
            "abs_belief": abs_belief,
            "abs_stoch": read_mask * prior_abs_stoch + copy_mask * prev_state["abs_stoch"],
            "abs_mean": prior_abs_stats["mean"],
            "abs_std": prior_abs_stats["std"],
            "obs_belief": obs_belief,
            "obs_stoch": prior_obs_stoch,
            "obs_mean": prior_obs_stats["mean"],
            "obs_std": prior_obs_stats["std"],
            "boundary": read_mask,
            "boundary_logit": prior_boundary_logit_reg,  # prior boundary logit (regularized)
            "seg_len": seg_len,
            "seg_num": seg_num,
        }
        
        return post, prior
    
    def img_step(self, prev_state, prev_action, sample=True, boundary_mode="prior"):
        """
        Single imagination step (no observation).
        
        Args:
            prev_state: previous hierarchical state
            prev_action: action to take
            sample: whether to sample or use mode
            boundary_mode: 'prior', 'fixed', or 'none'
            
        Returns:
            prior: prior state after transition
        """
        # Boundary decision during imagination
        obs_feat = self._get_obs_feat(prev_state)
        boundary_logit = self.prior_boundary(obs_feat)
        boundary_logit = self._regularize_boundary(
            boundary_logit, prev_state["seg_len"], prev_state["seg_num"]
        )
        
        if boundary_mode == "prior":
            boundary_sample, _ = self._sample_boundary(boundary_logit)
        elif boundary_mode == "fixed":
            # Fixed interval: boundary every max_seg_len steps
            is_boundary = (prev_state["seg_len"] >= self._max_seg_len - 1).float()
            boundary_sample = torch.stack([is_boundary, 1 - is_boundary], dim=-1).squeeze(-2)
        else:  # "none"
            # No boundaries during imagination
            boundary_sample = torch.zeros_like(boundary_logit)
            boundary_sample[:, 1] = 1.0  # Always COPY
        
        read_mask = boundary_sample[:, 0:1]
        copy_mask = boundary_sample[:, 1:2]
        
        # Update segment tracking
        seg_len = read_mask * 1.0 + copy_mask * (prev_state["seg_len"] + 1.0)
        seg_num = read_mask * (prev_state["seg_num"] + 1.0) + copy_mask * prev_state["seg_num"]
        
        # Abstract level
        abs_inp = self._abs_inp_layers(torch.cat([prev_state["abs_stoch"], prev_action], dim=-1))
        abs_belief_updated, _ = self._abs_cell(abs_inp, [prev_state["abs_belief"]])
        abs_belief = read_mask * abs_belief_updated + copy_mask * prev_state["abs_belief"]
        
        prior_abs_stats = self.prior_abs_state(abs_belief)
        prior_abs_dist = self.prior_abs_state.get_dist(prior_abs_stats)
        
        if sample:
            abs_stoch_new = prior_abs_dist.rsample()
        else:
            abs_stoch_new = prior_abs_dist.mean
        
        abs_stoch = read_mask * abs_stoch_new + copy_mask * prev_state["abs_stoch"]
        abs_feat = torch.cat([abs_belief, abs_stoch], dim=-1)
        
        # Observation level
        obs_inp = self._obs_inp_layers(torch.cat([prev_state["obs_stoch"], abs_feat], dim=-1))
        obs_belief_updated, _ = self._obs_cell(obs_inp, [prev_state["obs_belief"]])
        obs_belief_init = self._init_obs_belief(abs_feat)
        obs_belief = read_mask * obs_belief_init + copy_mask * obs_belief_updated
        
        prior_obs_stats = self.prior_obs_state(obs_belief)
        prior_obs_dist = self.prior_obs_state.get_dist(prior_obs_stats)
        
        if sample:
            obs_stoch = prior_obs_dist.rsample()
        else:
            obs_stoch = prior_obs_dist.mean
        
        return {
            "abs_belief": abs_belief,
            "abs_stoch": abs_stoch,
            "abs_mean": prior_abs_stats["mean"],
            "abs_std": prior_abs_stats["std"],
            "obs_belief": obs_belief,
            "obs_stoch": obs_stoch,
            "obs_mean": prior_obs_stats["mean"],
            "obs_std": prior_obs_stats["std"],
            "boundary": read_mask,
            "boundary_logit": boundary_logit,
            "seg_len": seg_len,
            "seg_num": seg_num,
        }
    
    def jumpy_img_step(self, prev_state, prev_action, sample=True):
        """
        Jumpy imagination step - abstract level only.
        Each step represents a full segment (jump to next boundary).
        
        Args:
            prev_state: previous state
            prev_action: action to take
            sample: whether to sample or use mode
            
        Returns:
            prior: prior state after abstract transition
        """
        # Always assume boundary (each step is a new segment)
        read_mask = torch.ones(prev_state["abs_belief"].shape[0], 1, device=self._device)
        
        # Abstract level transition
        abs_inp = self._abs_inp_layers(torch.cat([prev_state["abs_stoch"], prev_action], dim=-1))
        abs_belief, _ = self._abs_cell(abs_inp, [prev_state["abs_belief"]])
        
        prior_abs_stats = self.prior_abs_state(abs_belief)
        prior_abs_dist = self.prior_abs_state.get_dist(prior_abs_stats)
        
        if sample:
            abs_stoch = prior_abs_dist.rsample()
        else:
            abs_stoch = prior_abs_dist.mean
        
        abs_feat = torch.cat([abs_belief, abs_stoch], dim=-1)
        
        # Initialize observation level for new segment
        obs_belief = self._init_obs_belief(abs_feat)
        
        prior_obs_stats = self.prior_obs_state(obs_belief)
        prior_obs_dist = self.prior_obs_state.get_dist(prior_obs_stats)
        
        if sample:
            obs_stoch = prior_obs_dist.rsample()
        else:
            obs_stoch = prior_obs_dist.mean
        
        return {
            "abs_belief": abs_belief,
            "abs_stoch": abs_stoch,
            "abs_mean": prior_abs_stats["mean"],
            "abs_std": prior_abs_stats["std"],
            "obs_belief": obs_belief,
            "obs_stoch": obs_stoch,
            "obs_mean": prior_obs_stats["mean"],
            "obs_std": prior_obs_stats["std"],
            "boundary": read_mask,
            "boundary_logit": torch.zeros(read_mask.shape[0], 2, device=self._device),
            "seg_len": torch.ones_like(read_mask),
            "seg_num": prev_state["seg_num"] + 1,
        }
    
    def imagine_with_action(self, action, state, jumpy=False, boundary_mode="prior"):
        """
        Imagine trajectory given actions.
        
        Args:
            action: (batch, time, action_size) actions
            state: initial state
            jumpy: if True, use jumpy imagination (abstract only)
            boundary_mode: 'prior', 'fixed', or 'none'
            
        Returns:
            prior: states over time
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)  # (time, batch, action)
        
        priors = {k: [state[k]] for k in state.keys()}
        
        for t in range(action.shape[0]):
            if jumpy:
                state = self.jumpy_img_step(state, action[t])
            else:
                state = self.img_step(state, action[t], boundary_mode=boundary_mode)
            
            for k in state.keys():
                priors[k].append(state[k])
        
        # Stack and swap back
        priors = {k: swap(torch.stack(v, dim=0)) for k, v in priors.items()}
        return priors
    
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale, mask_scale=1.0):
        """
        Compute KL divergence losses for hierarchical model.
        
        Returns:
            loss: total KL loss (batch, time)
            value: KL value for logging
            dyn_loss: dynamics loss
            rep_loss: representation loss
            kl_mask: boundary KL loss (for mask encoder learning)
        """
        kld = torchd.kl.kl_divergence
        
        # Abstract level KL - detach parameters for prior
        post_abs_dist = torchd.normal.Normal(post["abs_mean"], post["abs_std"])
        prior_abs_dist_sg = torchd.normal.Normal(prior["abs_mean"].detach(), prior["abs_std"].detach())
        abs_kl = kld(
            torchd.independent.Independent(post_abs_dist, 1),
            torchd.independent.Independent(prior_abs_dist_sg, 1),
        )
        
        # Observation level KL - detach parameters for prior
        post_obs_dist = torchd.normal.Normal(post["obs_mean"], post["obs_std"])
        prior_obs_dist_sg = torchd.normal.Normal(prior["obs_mean"].detach(), prior["obs_std"].detach())
        obs_kl = kld(
            torchd.independent.Independent(post_obs_dist, 1),
            torchd.independent.Independent(prior_obs_dist_sg, 1),
        )
        
        # Combined KL with boundary weighting
        # KL is more important at boundaries for abstract level
        boundary = post["boundary"]
        kl_value = boundary.squeeze(-1) * abs_kl + obs_kl
        
        # Apply free bits
        rep_loss = torch.clip(kl_value, min=free)
        
        # Dynamics loss (prior learning) - detach parameters for posterior
        post_abs_dist_sg = torchd.normal.Normal(post["abs_mean"].detach(), post["abs_std"].detach())
        prior_abs_dist = torchd.normal.Normal(prior["abs_mean"], prior["abs_std"])
        dyn_abs_kl = kld(
            torchd.independent.Independent(post_abs_dist_sg, 1),
            torchd.independent.Independent(prior_abs_dist, 1),
        )
        post_obs_dist_sg = torchd.normal.Normal(post["obs_mean"].detach(), post["obs_std"].detach())
        prior_obs_dist = torchd.normal.Normal(prior["obs_mean"], prior["obs_std"])
        dyn_obs_kl = kld(
            torchd.independent.Independent(post_obs_dist_sg, 1),
            torchd.independent.Independent(prior_obs_dist, 1),
        )
        dyn_loss = boundary.squeeze(-1) * dyn_abs_kl + dyn_obs_kl
        dyn_loss = torch.clip(dyn_loss, min=free)
        
        # ========================
        # KL Mask: Boundary KL divergence
        # ========================
        # KL between posterior boundary and prior boundary distributions
        # This trains the prior boundary detector to match the posterior
        # KL(q(b|x_{1:T}) || p(b|s_{t-1}))
        post_boundary_logit = post["boundary_logit"]  # (batch, time, 2) or (batch, 2)
        prior_boundary_logit = prior["boundary_logit"]  # (batch, time, 2) or (batch, 2)
        
        # Convert logits to probabilities (softmax)
        post_boundary_prob = F.softmax(post_boundary_logit, dim=-1)
        prior_boundary_prob = F.softmax(prior_boundary_logit.detach(), dim=-1)  # detach prior for rep_loss
        
        # KL divergence for categorical: sum(p * log(p/q))
        eps = 1e-8
        kl_mask = (post_boundary_prob * (torch.log(post_boundary_prob + eps) - torch.log(prior_boundary_prob + eps))).sum(dim=-1)
        
        # Also compute dynamics version (prior learning)
        prior_boundary_prob_dyn = F.softmax(prior_boundary_logit, dim=-1)
        post_boundary_prob_sg = F.softmax(post_boundary_logit.detach(), dim=-1)  # detach post for dyn_loss
        kl_mask_dyn = (post_boundary_prob_sg * (torch.log(post_boundary_prob_sg + eps) - torch.log(prior_boundary_prob_dyn + eps))).sum(dim=-1)
        
        # Add mask KL to losses
        # mask_scale controls the weight of the boundary KL term
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss + mask_scale * kl_mask_dyn
        
        return loss, kl_value, dyn_loss, rep_loss, kl_mask


class GRUCell(nn.Module):
    """GRU Cell with LayerNorm (same as in networks.py)."""
    
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super().__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))
    
    @property
    def state_size(self):
        return self._size
    
    def forward(self, inputs, state):
        state = state[0]
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
