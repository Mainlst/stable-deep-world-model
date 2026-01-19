"""
Goal Autoencoder for Director-style Hierarchical Policy Learning

Based on "Deep Hierarchical Planning from Pixels" (Director)
https://arxiv.org/abs/2206.04114

The Goal Autoencoder compresses world model states into discrete latent codes
that serve as subgoals for the hierarchical policy.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from . import tools


class GoalAutoencoder(nn.Module):
    """
    Goal Autoencoder that compresses obs_feat into discrete latent codes.
    
    The encoder maps obs_feat to a discrete categorical distribution.
    The decoder reconstructs obs_feat from the discrete latent code.
    
    Loss: reconstruction_loss + beta * KL(encoder || uniform_prior)
    """
    
    def __init__(
        self,
        feat_size,
        hidden=512,
        layers=4,
        stoch=8,
        discrete=8,
        act="SiLU",
        norm=True,
        kl_config=None,  # Replaces kl_beta
        unimix_ratio=0.01,
        context_size=0,  # kept for compatibility but should be 0
        device=None,
        dist="mse",
    ):
        """
        Args:
            feat_size: Size of input features (obs_feat dimension)
            hidden: Hidden layer size
            layers: Number of hidden layers
            stoch: Number of categorical variables
            discrete: Number of categories per variable
            act: Activation function
            norm: Whether to use layer normalization
            kl_config: Dictionary for AutoAdapt configuration
            unimix_ratio: Uniform mixing ratio for encoder output
            context_size: Unused, kept for API compatibility.
            device: Device to use
            dist: Type of distribution for reconstruction ('mse' or 'symlog_mse')
        """
        super().__init__()
        self._feat_size = feat_size
        self._hidden = hidden
        self._stoch = stoch
        self._discrete = discrete
        self._kl_config = kl_config or {'impl': 'fixed', 'scale': 1.0}
        self._unimix_ratio = unimix_ratio
        
        # Initialize AutoAdapt for dynamic KL scaling
        self._kl_adapt = tools.AutoAdapt((), **self._kl_config, device=device)
        self._context_size = 0  # Force 0 as official implementation does not use context
        self._device = device
        self._dist = dist
        
        act_fn = getattr(torch.nn, act)
        
        # Encoder: goal -> discrete latent logits
        # FIXED: Encoder takes ONLY goal, consistent with official Director
        enc_layers = []
        inp_dim = feat_size
        for i in range(layers):
            enc_layers.append(nn.Linear(inp_dim, hidden, bias=not norm))
            if norm:
                enc_layers.append(nn.LayerNorm(hidden, eps=1e-03))
            enc_layers.append(act_fn())
            inp_dim = hidden
        enc_layers.append(nn.Linear(hidden, stoch * discrete))
        self.encoder = nn.Sequential(*enc_layers)
        self.encoder.apply(tools.weight_init)
        self.encoder[-1].apply(tools.uniform_weight_init(1.0))
        
        # Decoder: discrete latent -> reconstructed feat
        # FIXED: Decoder takes ONLY discrete latent, consistent with official Director
        dec_layers = []
        inp_dim = stoch * discrete
        for i in range(layers):
            dec_layers.append(nn.Linear(inp_dim, hidden, bias=not norm))
            if norm:
                dec_layers.append(nn.LayerNorm(hidden, eps=1e-03))
            dec_layers.append(act_fn())
            inp_dim = hidden
        dec_layers.append(nn.Linear(hidden, feat_size))
        self.decoder = nn.Sequential(*dec_layers)
        self.decoder.apply(tools.weight_init)
        # FIXED: Decoder takes valid outscale=0.1
        self.decoder[-1].apply(tools.uniform_weight_init(0.1))

    @property
    def latent_size(self):
        """Size of flattened latent representation."""
        return self._stoch * self._discrete
    
    def encode(self, feat, context=None, sample=True):
        """
        Encode features to discrete latent.
        
        Args:
            feat: (..., feat_size) input features (goal)
            context: Ignored (kept for API compatibility)
            sample: Whether to sample or use mode
            
        Returns:
            z: (..., stoch, discrete) one-hot encoded latent
            dist: Distribution object
        """
        # FIXED: Do not use context
        enc_input = feat
        
        logits = self.encoder(enc_input)
        logits = logits.reshape(feat.shape[:-1] + (self._stoch, self._discrete))
        
        # Apply unimix (uniform mixing for exploration)
        if self._unimix_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self._discrete
            probs = (1 - self._unimix_ratio) * probs + self._unimix_ratio * uniform
            logits = torch.log(probs + 1e-8)
        
        dist = tools.OneHotDist(logits, unimix_ratio=0.0)  # Already applied unimix
        
        if sample:
            z = dist.sample()
        else:
            z = dist.mode()
        
        return z, dist
    
    def decode(self, z, context=None):
        """
        Decode discrete latent to reconstructed features.
        
        Args:
            z: (..., stoch, discrete) one-hot encoded latent
            context: Ignored (kept for API compatibility)
            
        Returns:
            dist: Distribution over reconstructed features (MSEDist or SymlogDist)
        """
        # Flatten latent
        z_flat = z.reshape(z.shape[:-2] + (self._stoch * self._discrete,))
        
        # FIXED: Do not use context
        decoder_input = z_flat
        
        out = self.decoder(decoder_input)
        
        if self._dist == 'mse':
            return tools.MSEDist(out)
        elif self._dist == 'symlog_mse':
            return tools.SymlogDist(out)
        else:
            return out
    
    def loss(self, feat, context=None):
        """
        Compute Goal AE loss.
        
        Args:
            feat: (..., feat_size) input features (goal)
            context: (..., context_size) context features for conditional encoding/decoding
            
        Returns:
            loss: Scalar loss value
            metrics: Dict of metrics
        """
        # Encode with context (CRITICAL FIX 4)
        z, dist = self.encode(feat, context=context, sample=True)
        
        # Decode with context
        recon_dist = self.decode(z, context=context)
        
        # Reconstruction loss (Negative Log Likelihood)
        # Official: rec = -dec.log_prob(tf.stop_gradient(goal))
        recon_loss = -recon_dist.log_prob(feat.detach())
        
        # KL divergence against uniform prior
        # Get logits from distribution
        probs = dist.distribution.probs if hasattr(dist, 'distribution') else F.softmax(dist.logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).sum(-1)  # Sum over discrete and stoch
        log_k = torch.log(torch.tensor(self._discrete, dtype=feat.dtype, device=feat.device))
        kl = -entropy + self._stoch * log_k  # KL against uniform
        
        # Total loss
        # Use AutoAdapt for KL scaling
        kl_loss, kl_mets = self._kl_adapt(kl)
        
        loss = recon_loss + kl_loss
        
        metrics = {
            "goal_ae_recon": recon_loss.mean().detach(),
            "goal_ae_kl": kl.mean().detach(),
            "goal_ae_loss": loss.mean().detach(),
        }
        metrics.update({f'goalkl_{k}': v.detach() for k, v in kl_mets.items()})
        
        return loss.mean(), metrics
    
    def forward(self, feat, sample=True):
        """
        Forward pass: encode and decode.
        
        Args:
            feat: Input features
            sample: Whether to sample latent
            
        Returns:
            recon: Reconstructed features
            z: Latent code
            dist: Encoder distribution
        """
        z, dist = self.encode(feat, sample=sample)
        recon = self.decode(z)
        return recon, z, dist


class GoalEncoder(nn.Module):
    """
    Standalone Goal Encoder for Manager policy.
    
    This is used by the Manager to encode current state into a goal latent,
    which is then decoded by the Goal Decoder to produce a subgoal.
    """
    
    def __init__(
        self,
        input_size,
        hidden=512,
        layers=4,
        stoch=8,
        discrete=8,
        act="SiLU",
        norm=True,
        unimix_ratio=0.01,
        device=None,
    ):
        super().__init__()
        self._stoch = stoch
        self._discrete = discrete
        self._unimix_ratio = unimix_ratio
        
        act_fn = getattr(torch.nn, act)
        
        enc_layers = []
        inp_dim = input_size
        for i in range(layers):
            enc_layers.append(nn.Linear(inp_dim, hidden, bias=not norm))
            if norm:
                enc_layers.append(nn.LayerNorm(hidden, eps=1e-03))
            enc_layers.append(act_fn())
            inp_dim = hidden
        enc_layers.append(nn.Linear(hidden, stoch * discrete))
        self.layers = nn.Sequential(*enc_layers)
        self.layers.apply(tools.weight_init)
        self.layers[-1].apply(tools.uniform_weight_init(1.0))
    
    def forward(self, x):
        """
        Encode input to discrete distribution.
        
        Args:
            x: (..., input_size) input features
            
        Returns:
            dist: OneHotDist over (stoch, discrete)
        """
        logits = self.layers(x)
        logits = logits.reshape(x.shape[:-1] + (self._stoch, self._discrete))
        return tools.OneHotDist(logits, unimix_ratio=self._unimix_ratio)


class GoalDecoder(nn.Module):
    """
    Standalone Goal Decoder.
    
    Decodes discrete latent code (skill) to a goal in obs_feat space.
    Optionally takes context for conditional decoding.
    """
    
    def __init__(
        self,
        output_size,
        hidden=512,
        layers=4,
        stoch=8,
        discrete=8,
        act="SiLU",
        norm=True,
        use_context=False,
        context_size=0,
        device=None,
    ):
        super().__init__()
        self._stoch = stoch
        self._discrete = discrete
        
        act_fn = getattr(torch.nn, act)
        
        inp_dim = stoch * discrete
        if use_context:
            inp_dim += context_size
        
        dec_layers = []
        for i in range(layers):
            dec_layers.append(nn.Linear(inp_dim, hidden, bias=not norm))
            if norm:
                dec_layers.append(nn.LayerNorm(hidden, eps=1e-03))
            dec_layers.append(act_fn())
            inp_dim = hidden
        dec_layers.append(nn.Linear(hidden, output_size))
        self.layers = nn.Sequential(*dec_layers)
        self.layers.apply(tools.weight_init)
        self.layers[-1].apply(tools.uniform_weight_init(1.0))
        
        self._use_context = use_context
    
    def forward(self, skill, context=None):
        """
        Decode skill to goal.
        
        Args:
            skill: (..., stoch, discrete) one-hot skill
            context: (..., context_size) optional context
            
        Returns:
            goal: (..., output_size) decoded goal
        """
        # Flatten skill
        skill_flat = skill.reshape(skill.shape[:-2] + (self._stoch * self._discrete,))
        
        if self._use_context and context is not None:
            x = torch.cat([skill_flat, context], dim=-1)
        else:
            x = skill_flat
        
        return self.layers(x)
