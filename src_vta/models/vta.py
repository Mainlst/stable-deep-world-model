"""Top-level VTA model wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from .rssm import HierarchicalRSSM


class VTA(nn.Module):
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
        self.belief_size = belief_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        self.state_model = HierarchicalRSSM(
            belief_size=self.belief_size,
            state_size=self.state_size,
            act_size=act_size,
            num_layers=self.num_layers,
            max_seg_len=self.max_seg_len,
            max_seg_num=self.max_seg_num,
            loss_type=loss_type,
        )

    def forward(self, obs_data_list, act_data_list, seq_size, init_size, obs_std=1.0, loss_type="bce"):
        [
            obs_rec_list,
            prior_boundary_log_density_list,
            post_boundary_log_density_list,
            prior_abs_state_list,
            post_abs_state_list,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list,
        ] = self.state_model(obs_data_list, act_data_list, seq_size, init_size)

        obs_target_list = obs_data_list[:, init_size:-init_size]
        if loss_type == "bce":
            # BCEはautocast混合精度で不安定になるためFP32で計算
            with torch.cuda.amp.autocast(enabled=False):
                obs_cost = F.binary_cross_entropy(
                    obs_rec_list.float(), obs_target_list.float(), reduction="none"
                )
        else:
            obs_cost = -Normal(obs_rec_list, obs_std).log_prob(obs_target_list)
        obs_cost = obs_cost.sum(dim=[2, 3, 4])

        kl_abs_state_list = []
        kl_obs_state_list = []
        for t in range(seq_size):
            read_data = boundary_data_list[:, t].detach()
            kl_abs_state = kl_divergence(post_abs_state_list[t], prior_abs_state_list[t]) * read_data
            kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
            kl_abs_state_list.append(kl_abs_state.sum(-1))
            kl_obs_state_list.append(kl_obs_state.sum(-1))
        kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)
        kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

        kl_mask_list = (post_boundary_log_density_list - prior_boundary_log_density_list)

        return {
            "rec_data": obs_rec_list,
            "mask_data": boundary_data_list,
            "obs_cost": obs_cost,
            "kl_abs_state": kl_abs_state_list,
            "kl_obs_state": kl_obs_state_list,
            "kl_mask": kl_mask_list,
            "p_mask": prior_boundary_list.mean,
            "q_mask": post_boundary_list.mean,
            "p_ent": prior_boundary_list.entropy(),
            "q_ent": post_boundary_list.entropy(),
            "beta": self.state_model.mask_beta,
            "train_loss": obs_cost.mean() + kl_abs_state_list.mean() + kl_obs_state_list.mean() + kl_mask_list.mean(),
        }

    def jumpy_generation(self, *args):
        return self.state_model.jumpy_generation(*args)

    def full_generation(self, *args):
        return self.state_model.full_generation(*args)


__all__ = ["VTA"]
