import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

# densy.py, action.py をインポート
from .action import ActionModel
from .dense import DenseModel

from config import Config

class DreamerAgent(nn.Module):
    def __init__(self, vta_model, config: Config):
        super().__init__()
        self.vta = vta_model
        self.cfg = config
        
        # VTAのパラメータ
        feat_size = self.vta.state_model.abs_feat_size
        act_size = self.vta.state_model.act_size
        
        # --- Behavior Models ---
        # Configから分布タイプを決定
        if self.cfg.action_size == "discrete":
            action_dist_type = "onehot"
        else:
            action_dist_type = "tanh_normal"

        self.actor = ActionModel(
            act_size, 
            feat_size, 
            dist=action_dist_type, # ★変更: Configに合わせて切り替え
            mean_scale=5.0
        )
        self.value = DenseModel(feat_size, (1,))
        
        # Reward Model
        self.reward_model = DenseModel(feat_size, (1,))
        
        # Discount Model: 継続確率(1=continue, 0=done)を予測
        # dist="binary" (Bernoulli) なので出力は確率分布
        self.discount_model = DenseModel(feat_size, (1,), dist="binary")

    def forward(self, obs, actions, rewards, dones, seq_len, init_len):
        """
        Args:
            obs: (B, T, C, H, W)
            actions: (B, T, ActDim)
            rewards: (B, T, 1)
            dones: (B, T, 1) - 環境からの終了フラグ (1=終了, 0=継続)
        """
        # 1. World Model Learning (VTA Forward)
        vta_outputs = self.vta(obs, actions, seq_len, init_len)
        vta_loss = vta_outputs["train_loss"]
        
        # 2. Behavior Learning
        actor_loss, value_loss, reward_loss, discount_loss, entropy = self._behavior_learning(
            vta_outputs, actions, rewards, dones
        )
        
        # Total Loss
        total_loss = vta_loss + actor_loss + value_loss + reward_loss + discount_loss
        
        logs = {
            "vta_loss": vta_loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "reward_loss": reward_loss.item(),
            "discount_loss": discount_loss.item(),
            "entropy": entropy.item(),
            "train_loss": total_loss.item()
        }
        
        return total_loss, logs

    def _behavior_learning(self, vta_outputs, true_actions, true_rewards, true_dones):
        # データの取得 (勾配を切る)
        post_states_list = vta_outputs["post_abs_state_list"] 
        post_beliefs_list = vta_outputs["post_abs_belief_list"]
        mask_data = vta_outputs["mask_data"].detach()
        
        # -----------------------------------------------------------
        # A. Models Training (Reward & Discount)
        # -----------------------------------------------------------
        
        # 特徴量の準備
        states = torch.stack([d.rsample() for d in post_states_list], dim=1).detach()
        beliefs = torch.stack(post_beliefs_list, dim=1).detach()
        features = self.vta.state_model.abs_feat(
            torch.cat([beliefs, states], dim=-1)
        )

        # --- Reward Model Training ---
        target_rewards = self._compute_segment_rewards(true_rewards, mask_data, self.cfg.init_size)
        pred_reward_dist_sup = self.reward_model(features)
        reward_loss = -pred_reward_dist_sup.log_prob(target_rewards).mean()
        reward_loss = self.cfg.loss_scale_reward * reward_loss

        # --- Discount Model Training ---
        # セグメント内で一度でもdoneになれば、そのセグメントは「終了」とみなす
        target_dones = self._compute_segment_dones(true_dones, mask_data, self.cfg.init_size)
        # Discount Modelは「継続確率(1.0=継続)」を予測するため反転させる
        target_discount = (1.0 - target_dones).float()
        
        pred_discount_dist_sup = self.discount_model(features)
        discount_loss = -pred_discount_dist_sup.log_prob(target_discount).mean()
        
        # Configに loss_scale_discount があれば使い、なければ1.0とする
        scale_discount = getattr(self.cfg, "loss_scale_discount", 1.0)
        discount_loss = scale_discount * discount_loss

        # -----------------------------------------------------------
        # B. Dreamer Actor-Critic (Imagination)
        # -----------------------------------------------------------
        
        with torch.no_grad():
            start_states = states.view(-1, states.shape[-1])
            start_beliefs = beliefs.view(-1, beliefs.shape[-1])
        
        # Imagine
        horizon = self.cfg.horizon
        imag_feat, imag_action = self.vta.state_model.imagine_abstract(
            start_states, start_beliefs, self.actor, horizon
        )
        
        # 予測
        pred_reward_dist = self.reward_model(imag_feat)
        pred_reward = pred_reward_dist.mean
        pred_value_dist = self.value(imag_feat)
        pred_values = pred_value_dist.mean
        
        # 予測された継続確率を使用する
        pred_discount_dist = self.discount_model(imag_feat)
        pred_discount = pred_discount_dist.mean

        # Lambda-target計算
        target_value = self._lambda_return(
            pred_reward, pred_values, pred_discount, 
            lambda_=self.cfg.return_lambda, 
            gamma=self.cfg.gamma
        )
        
        # --- Actor Loss ---
        current_action_dist = self.actor(imag_feat)
        entropy = current_action_dist.entropy().mean()
        
        actor_loss = -torch.mean(target_value)
        if self.cfg.actor_entropy_scale > 0:
            actor_loss -= self.cfg.actor_entropy_scale * entropy
        actor_loss = self.cfg.loss_scale_actor * actor_loss
        
        # --- Value Loss ---
        value_loss = 0.5 * torch.mean((pred_values - target_value.detach()) ** 2)
        value_loss = self.cfg.loss_scale_value * value_loss
        
        return actor_loss, value_loss, reward_loss, discount_loss, entropy

    def _compute_segment_rewards(self, rewards, mask, init_size):
        # 累積和(Sum)による集約
        seq_rewards = rewards[:, init_size : init_size + mask.shape[1]]
        target_rewards = torch.zeros_like(seq_rewards)
        accumulated = torch.zeros_like(seq_rewards[:, 0])
        
        for t in reversed(range(seq_rewards.shape[1])):
            r = seq_rewards[:, t]
            m = mask[:, t]
            accumulated = accumulated + r
            target_rewards[:, t] = accumulated
            accumulated = accumulated * (1.0 - m)
            
        return target_rewards.detach()

    def _compute_segment_dones(self, dones, mask, init_size):
        """
        環境のdoneフラグ(dones)を、VTAのセグメント境界(mask)に合わせて集約する。
        セグメント期間中に一度でもdone=1があれば、そのセグメント全体を「終了」とする。
        """
        # dones: (B, Full_T, 1) -> VTA出力サイズへ
        seq_dones = dones[:, init_size : init_size + mask.shape[1]]
        target_dones = torch.zeros_like(seq_dones)
        
        # 論理和(Max)による集約
        # 0.0 or 1.0 なので max で論理和の代用が可能
        accumulated = torch.zeros_like(seq_dones[:, 0])
        
        for t in reversed(range(seq_dones.shape[1])):
            d = seq_dones[:, t]
            m = mask[:, t] # 1.0 if READ (New segment start)
            
            # 既存の累積値と現在のdoneの大きい方を取る (OR演算)
            accumulated = torch.max(accumulated, d)
            
            target_dones[:, t] = accumulated
            
            # READ=1ならリセット (次のセグメントへは持ち越さない)
            accumulated = accumulated * (1.0 - m)
            
        return target_dones.detach()

    def _lambda_return(self, reward, value, discount, lambda_, gamma):
        returns = torch.zeros_like(reward)
        next_value = value[:, -1]
        
        for t in reversed(range(reward.shape[1])):
            r = reward[:, t]
            d = discount[:, t]
            v = value[:, t]
            
            # inputs: r + gamma * discount * ...
            # discount予測が入ることで、終了予測地点より先の価値が遮断される
            inputs = r + gamma * d * (1 - lambda_) * next_value
            if t == reward.shape[1] - 1:
                returns[:, t] = inputs + gamma * d * lambda_ * next_value
            else:
                returns[:, t] = inputs + gamma * d * lambda_ * returns[:, t+1]
            
            next_value = v
            
        return returns.detach()