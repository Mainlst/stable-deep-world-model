"""Hierarchical RSSM core for VTA."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, kl_divergence

from .components import (
    Decoder,
    Encoder,
    LatentDistribution,
    PostBoundaryDetector,
    PriorBoundaryDetector,
)


class HierarchicalRSSM(nn.Module):
    """Implements the hierarchical recurrent state-space model used by VTA."""

    def __init__(
        self,
        belief_size,
        state_size,
        act_size,
        num_layers,
        max_seg_len,
        max_seg_num,
        loss_type="bce",
    ):
        super().__init__()
        # Sizes
        self.abs_belief_size = belief_size
        self.abs_state_size = state_size
        self.abs_feat_size = belief_size
        self.obs_belief_size = belief_size
        self.obs_state_size = state_size
        self.obs_feat_size = belief_size
        self.num_layers = num_layers
        self.feat_size = belief_size
        self.act_size = act_size
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        # Gumbel-Softmax temperature
        self.mask_beta = 1.0

        dec_activation = "sigmoid" if loss_type == "bce" else ("tanh" if loss_type == "mse" else "none")

        # Encoders / decoders
        self.enc_obs = Encoder(feat_size=self.feat_size)
        self.dec_obs = Decoder(input_size=self.obs_feat_size, feat_size=self.feat_size, activation=dec_activation)

        # Boundary detectors
        self.prior_boundary = PriorBoundaryDetector(self.obs_feat_size)
        self.post_boundary = PostBoundaryDetector(self.feat_size, num_layers=self.num_layers)

        # Feature extractors
        self.abs_feat = nn.Linear(self.abs_belief_size + self.abs_state_size, self.abs_feat_size)
        self.obs_feat = nn.Linear(self.obs_belief_size + self.obs_state_size, self.obs_feat_size)

        # Belief initializers
        self.init_abs_belief = nn.Identity()
        self.init_obs_belief = nn.Identity()

        # Belief updates
        self.update_abs_belief = nn.GRUCell(self.abs_state_size + self.act_size, self.abs_belief_size)
        self.update_obs_belief = nn.GRUCell(self.obs_state_size + self.abs_feat_size, self.obs_belief_size)

        # Posterior encoders (bi-directional style)
        self.abs_post_fwd = nn.GRUCell(self.feat_size, self.abs_belief_size)
        self.abs_post_bwd = nn.GRUCell(self.feat_size, self.abs_belief_size)
        self.obs_post_fwd = nn.GRUCell(self.feat_size, self.obs_belief_size)

        # Priors
        self.prior_abs_state = LatentDistribution(input_size=self.abs_belief_size, latent_size=self.abs_state_size)
        self.prior_obs_state = LatentDistribution(input_size=self.obs_belief_size, latent_size=self.obs_state_size)

        # Posteriors
        self.post_abs_state = LatentDistribution(
            input_size=self.abs_belief_size + self.abs_belief_size, latent_size=self.abs_state_size
        )
        self.post_obs_state = LatentDistribution(
            input_size=self.obs_belief_size + self.abs_feat_size, latent_size=self.obs_state_size
        )

    @staticmethod
    def gumbel_sampling(log_alpha, temp, margin=1e-4):
        noise = log_alpha.new_empty(log_alpha.shape).uniform_(margin, 1 - margin)
        gumbel_sample = -torch.log(-torch.log(noise))
        return torch.div(log_alpha + gumbel_sample, temp)

    def boundary_sampler(self, log_alpha):
        """Sample boundary mask with straight-through Gumbel-Softmax."""
        if self.training:
            log_sample_alpha = self.gumbel_sampling(log_alpha=log_alpha, temp=self.mask_beta)
        else:
            log_sample_alpha = log_alpha / self.mask_beta
        log_sample_alpha = log_sample_alpha - torch.logsumexp(log_sample_alpha, dim=-1, keepdim=True)
        sample_prob = log_sample_alpha.exp()
        sample_data = torch.eye(2, dtype=log_alpha.dtype, device=log_alpha.device)[torch.max(sample_prob, dim=-1)[1]]
        sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())
        return sample_data, log_sample_alpha

    def regularize_prior_boundary(self, log_alpha_list, boundary_data_list):
        """Apply segment length/number constraints to the prior boundary logits."""
        if not self.training:
            return log_alpha_list

        num_samples = boundary_data_list.shape[0]
        seq_len = boundary_data_list.shape[1]
        seg_num = log_alpha_list.new_zeros(num_samples, 1)
        seg_len = log_alpha_list.new_zeros(num_samples, 1)

        one_prob = 1 - 1e-3
        max_scale = np.log(one_prob / (1 - one_prob))
        near_read_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_read_data[:, 1] = -near_read_data[:, 1]
        near_copy_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_copy_data[:, 0] = -near_copy_data[:, 0]

        new_log_alpha_list = []
        for t in range(seq_len):
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)
            seg_len = read_data * 1.0 + copy_data * (seg_len + 1.0)
            seg_num = read_data * (seg_num + 1.0) + copy_data * seg_num
            over_len = torch.ge(seg_len, self.max_seg_len).float().detach()
            over_num = torch.ge(seg_num, self.max_seg_num).float().detach()

            new_log_alpha = over_num * near_copy_data + (1.0 - over_num) * log_alpha_list[:, t]
            new_log_alpha = over_len * near_read_data + (1.0 - over_len) * new_log_alpha
            new_log_alpha_list.append(new_log_alpha)

        return torch.stack(new_log_alpha_list, dim=1)

    @staticmethod
    def log_density_concrete(log_alpha, log_sample, temp):
        exp_term = log_alpha - temp * log_sample
        log_prob = torch.sum(exp_term, -1) - 2.0 * torch.logsumexp(exp_term, -1)
        return log_prob

    def forward(self, obs_data_list: torch.Tensor, act_data_list: torch.Tensor, seq_size: int, init_size: int):
        num_samples, full_seq_size = obs_data_list.shape[:2]
        enc_obs_list = self.enc_obs(obs_data_list.view(-1, *obs_data_list.shape[2:]))
        enc_obs_list = enc_obs_list.view(num_samples, full_seq_size, -1)

        post_boundary_log_alpha_list = self.post_boundary(enc_obs_list)
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(post_boundary_log_alpha_list)

        boundary_data_list[:, :(init_size + 1), 0] = 1.0
        boundary_data_list[:, :(init_size + 1), 1] = 0.0
        boundary_data_list[:, -init_size:, 0] = 1.0
        boundary_data_list[:, -init_size:, 1] = 0.0

        abs_post_fwd_list = []
        abs_post_bwd_list = []
        obs_post_fwd_list = []
        abs_post_fwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_post_bwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        obs_post_fwd = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)
            abs_post_fwd = self.abs_post_fwd(enc_obs_list[:, fwd_t], abs_post_fwd)
            obs_post_fwd = self.obs_post_fwd(enc_obs_list[:, fwd_t], fwd_copy_data * obs_post_fwd)
            abs_post_fwd_list.append(abs_post_fwd)
            obs_post_fwd_list.append(obs_post_fwd)

            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            abs_post_bwd = self.abs_post_bwd(enc_obs_list[:, bwd_t], abs_post_bwd)
            abs_post_bwd_list.append(abs_post_bwd)
            abs_post_bwd = bwd_copy_data * abs_post_bwd
        abs_post_bwd_list = abs_post_bwd_list[::-1]

        obs_rec_list = []
        prior_abs_state_list = []
        post_abs_state_list = []
        post_abs_belief_list = [] # Belief保存用リスト
        prior_obs_state_list = []
        post_obs_state_list = []
        prior_boundary_log_alpha_list = []

        abs_belief = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = obs_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = obs_data_list.new_zeros(num_samples, self.obs_state_size)

        for t in range(init_size, init_size + seq_size):
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)

            if t == init_size:
                abs_belief = self.init_abs_belief(abs_post_fwd_list[t - 1])
            else:
                abs_belief = copy_data * abs_belief + read_data * self.update_abs_belief(
                    torch.cat([abs_state, act_data_list[:, t - 1]], dim=1),
                    abs_belief,
                )
                
            post_abs_belief_list.append(abs_belief) # Belief保存
            
            prior_abs_state = self.prior_abs_state(abs_belief)
            post_abs_state = self.post_abs_state(torch.cat([abs_post_fwd_list[t - 1], abs_post_bwd_list[t]], dim=1))
            abs_state = copy_data * abs_state + read_data * post_abs_state.rsample()
            abs_feat = self.abs_feat(torch.cat([abs_belief, abs_state], dim=1))

            obs_belief = copy_data * self.update_obs_belief(torch.cat([obs_state, abs_feat], dim=1), obs_belief) + \
                read_data * self.init_obs_belief(abs_feat)
            prior_obs_state = self.prior_obs_state(obs_belief)
            post_obs_state = self.post_obs_state(torch.cat([obs_post_fwd_list[t], abs_feat], dim=1))
            obs_state = post_obs_state.rsample()
            obs_feat = self.obs_feat(torch.cat([obs_belief, obs_state], dim=1))
            obs_rec_list.append(obs_feat)

            prior_boundary_log_alpha = self.prior_boundary(obs_feat)

            prior_boundary_log_alpha_list.append(prior_boundary_log_alpha)
            prior_abs_state_list.append(prior_abs_state)
            post_abs_state_list.append(post_abs_state)
            prior_obs_state_list.append(prior_obs_state)
            post_obs_state_list.append(post_obs_state)

        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        obs_rec_list = self.dec_obs(obs_rec_list.view(num_samples * seq_size, -1))
        obs_rec_list = obs_rec_list.view(num_samples, seq_size, *obs_rec_list.shape[-3:])

        prior_boundary_log_alpha_list = torch.stack(prior_boundary_log_alpha_list, dim=1)
        boundary_data_list = boundary_data_list[:, init_size:(init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[:, (init_size + 1):(init_size + 1 + seq_size)]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[:, (init_size + 1):(init_size + 1 + seq_size)]

        prior_boundary_log_alpha_list = self.regularize_prior_boundary(prior_boundary_log_alpha_list, boundary_data_list)

        prior_boundary_log_density = self.log_density_concrete(
            prior_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )
        post_boundary_log_density = self.log_density_concrete(
            post_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )

        prior_boundary_list = F.softmax(prior_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        post_boundary_list = F.softmax(post_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        prior_boundary_list = Bernoulli(probs=prior_boundary_list)
        post_boundary_list = Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)

        return [
            obs_rec_list,
            prior_boundary_log_density,
            post_boundary_log_density,
            prior_abs_state_list,
            post_abs_state_list,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list,
            post_abs_belief_list,
        ]

    def jumpy_generation(self, init_data_list, full_action_cond, seq_size):
        assert seq_size <= full_action_cond.shape[1] - init_data_list.shape[1], "Can't generate over the action conditions."
        self.eval()

        num_samples = init_data_list.shape[0]
        init_size = init_data_list.shape[1]
        abs_post_fwd = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        for t in range(init_size):
            abs_post_fwd = self.abs_post_fwd(self.enc_obs(init_data_list[:, t]), abs_post_fwd)

        abs_belief = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = init_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_rec_list = []

        for t in range(seq_size):
            if t == 0:
                abs_belief = self.init_abs_belief(abs_post_fwd)
            else:
                abs_belief = self.update_abs_belief(
                    torch.concat([abs_state, full_action_cond[:, init_size + t - 1]], dim=1),
                    abs_belief,
                )
            abs_state = self.prior_abs_state(abs_belief).rsample()
            abs_feat = self.abs_feat(torch.cat([abs_belief, abs_state], dim=1))

            obs_belief = self.init_obs_belief(abs_feat)
            obs_state = self.prior_obs_state(obs_belief).rsample()
            obs_feat = self.obs_feat(torch.cat([obs_belief, obs_state], dim=1))
            obs_rec = self.dec_obs(obs_feat)
            obs_rec_list.append(obs_rec)

        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        return obs_rec_list, None

    def full_generation(self, init_data_list, full_action_cond, seq_size):
        assert seq_size <= full_action_cond.shape[1] - init_data_list.shape[1], "Can't generate over the action conditions."
        self.eval()

        num_samples = init_data_list.shape[0]
        init_size = init_data_list.shape[1]
        abs_post_fwd = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        for t in range(init_size):
            abs_post_fwd = self.abs_post_fwd(self.enc_obs(init_data_list[:, t]), abs_post_fwd)

        abs_belief = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = init_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = init_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = init_data_list.new_zeros(num_samples, self.obs_state_size)

        obs_rec_list = []
        boundary_data_list = []
        read_data = init_data_list.new_ones(num_samples, 1)
        copy_data = 1 - read_data
        for t in range(seq_size):
            if t == 0:
                abs_belief = self.init_abs_belief(abs_post_fwd)
            else:
                abs_belief = read_data * self.update_abs_belief(
                    torch.concat([abs_state, full_action_cond[:, init_size + t - 1]], dim=1),
                    abs_belief,
                ) + copy_data * abs_belief
            abs_state = read_data * self.prior_abs_state(abs_belief).rsample() + copy_data * abs_state
            abs_feat = self.abs_feat(torch.cat([abs_belief, abs_state], dim=1))

            obs_belief = read_data * self.init_obs_belief(abs_feat) + \
                copy_data * self.update_obs_belief(torch.cat([obs_state, abs_feat], dim=1), obs_belief)
            obs_state = self.prior_obs_state(obs_belief).rsample()
            obs_feat = self.obs_feat(torch.cat([obs_belief, obs_state], dim=1))
            obs_rec = self.dec_obs(obs_feat)

            obs_rec_list.append(obs_rec)
            boundary_data_list.append(read_data)

            prior_boundary = self.boundary_sampler(self.prior_boundary(obs_feat))[0]
            read_data = prior_boundary[:, 0].unsqueeze(-1)
            copy_data = prior_boundary[:, 1].unsqueeze(-1)

        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        boundary_data_list = torch.stack(boundary_data_list, dim=1)
        return obs_rec_list, boundary_data_list
    
    def imagine_abstract(self, start_abs_state, start_abs_belief, actor, horizon):
        """
        Actorの行動に基づき、抽象レベルでの未来を予測する。
        m=1 (境界) のときはActorから行動を選択し、
        m=0 (継続) のときは前回の行動を継続（直進）する。
        境界(m)を予測するために、内部的にObsレベルの推論も行う。
        """
        device = start_abs_state.device
        num_samples = start_abs_state.shape[0]
        
        # 抽象状態の初期化
        abs_state = start_abs_state
        abs_belief = start_abs_belief
        
        # Obsレベル変数の初期化
        # 想像の開始点では観測レベルの状態を持っていないため、
        # 現在の抽象状態から生成されたと仮定して初期化する (read_data=1 相当)
        abs_feat_init = self.abs_feat(torch.cat([abs_belief, abs_state], dim=-1))
        obs_belief = self.init_obs_belief(abs_feat_init)
        obs_state = self.prior_obs_state(obs_belief).rsample()
        
        # 初期状態では「直前の境界検知によりリセットされた」とみなす
        read_data = torch.ones(num_samples, 1).to(device)
        copy_data = 1.0 - read_data
        
        # 行動の初期化（初回は必ずActorからサンプリングされるのでゼロでOK）
        prev_action = torch.zeros(num_samples, self.act_size).to(device)

        imag_feats = []
        imag_actions = []
        imag_states = []
        imag_beliefs = []
        
        for t in range(horizon):
            # 1. 現在の特徴量計算
            current_feat = self.abs_feat(torch.cat([abs_belief, abs_state], dim=-1))
            
            # 2. 行動選択
            # m=1 (read) の場合: Actorから新しい行動を選択
            # m=0 (copy) の場合: 前回の行動を維持（直進）
            action_dist = actor(current_feat)
            new_action = action_dist.rsample()
            
            # t=0 は強制的に新しい行動とする (初期read_data=1により自動的に処理される)
            action = read_data * new_action + copy_data * prev_action
            
            # 行動を更新
            prev_action = action

            # 3. 抽象状態 (Abs) の更新
            # input: state + action
            abs_input = torch.cat([abs_state, action], dim=-1)
            
            # m=1ならGRU更新、m=0ならBelief維持
            abs_belief = read_data * self.update_abs_belief(abs_input, abs_belief) + \
                         copy_data * abs_belief
            
            # State Sampling
            prior_dist = self.prior_abs_state(abs_belief)
            abs_state = read_data * prior_dist.rsample() + copy_data * abs_state
            
            # 4. 観測レベル (Obs) の更新 & 境界予測
            # 次のステップの m を決めるために Obs feat が必要
            
            # 更新されたAbs Stateに基づいて特徴量再計算
            abs_feat_next = self.abs_feat(torch.cat([abs_belief, abs_state], dim=-1))
            
            # Obs Belief Update
            # m=1: init(abs_feat), m=0: update(obs_state, abs_feat)
            obs_update_input = torch.cat([obs_state, abs_feat_next], dim=-1)
            obs_belief = read_data * self.init_obs_belief(abs_feat_next) + \
                         copy_data * self.update_obs_belief(obs_update_input, obs_belief)
            
            # Obs State Sampling
            prior_obs_dist = self.prior_obs_state(obs_belief)
            obs_state = prior_obs_dist.rsample()
            
            # Obs Feat & Boundary Prediction
            obs_feat = self.obs_feat(torch.cat([obs_belief, obs_state], dim=-1))
            prior_boundary_logits = self.prior_boundary(obs_feat)
            
            # 次のステップのための m をサンプリング
            boundary_sample, _ = self.boundary_sampler(prior_boundary_logits)
            read_data = boundary_sample[:, 0].unsqueeze(-1) # 0番目がREAD(1)
            copy_data = boundary_sample[:, 1].unsqueeze(-1) # 1番目がCOPY(0)
            
            # 保存
            imag_feats.append(current_feat)
            imag_actions.append(action)
            imag_states.append(abs_state)
            imag_beliefs.append(abs_belief)

        imag_feats = torch.stack(imag_feats, dim=1)
        imag_actions = torch.stack(imag_actions, dim=1)
        
        return imag_feats, imag_actions

__all__ = ["HierarchicalRSSM"]