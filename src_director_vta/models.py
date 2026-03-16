import copy
import torch
from torch import nn

from . import networks
from . import tools
from . import vta as vta_module

to_np = lambda x: x.detach().cpu().numpy()

import numpy as np



class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        # Choose dynamics model: RSSM or VTA
        self._dynamics_type = getattr(config, 'dynamics_type', 'rssm')
        if self._dynamics_type == 'vta':
            self.dynamics = vta_module.VTA(
                abs_belief=config.vta_abs_belief,
                abs_stoch=config.vta_abs_stoch,
                obs_belief=config.vta_obs_belief,
                obs_stoch=config.vta_obs_stoch,
                hidden=config.dyn_hidden,
                num_layers=config.dyn_rec_depth,
                max_seg_len=config.vta_max_seg_len,
                max_seg_num=config.vta_max_seg_num,
                boundary_temp=config.vta_boundary_temp,
                boundary_force_scale=config.vta_boundary_force_scale,
                act=config.act,
                norm=config.norm,
                min_std=config.dyn_min_std,
                num_actions=config.num_actions,
                embed_size=self.embed_size,
                device=config.device,
                vta_posterior_input=getattr(config, 'vta_posterior_input', 'embed'),
            )
            feat_size = self.dynamics.feat_size
        else:
            self.dynamics = networks.RSSM(
                config.dyn_stoch,
                config.dyn_deter,
                config.dyn_hidden,
                config.dyn_rec_depth,
                config.dyn_discrete,
                config.act,
                config.norm,
                config.dyn_mean_act,
                config.dyn_std_act,
                config.dyn_min_std,
                config.unimix_ratio,
                config.initial,
                config.num_actions,
                self.embed_size,
                config.device,
            )
            if config.dyn_discrete:
                feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            else:
                feat_size = config.dyn_stoch + config.dyn_deter
                
        self.goal_enc = networks.MLP(
            inp_dim=config.dyn_deter,  # TODO: 状態の次元, 今は決定論的状態を使用
            shape=(config.goal_enc_num_codebook, config.goal_enc_num_categorical),  
            layers=config.num_goal_enc_layers,  # GoalAEのMLP層数, default: 4
            units=config.goal_enc_hidden_units,  # GoalAEのMLPユニット数, default: 512
            act=config.goal_enc_act_fn,  # GoalAEの活性化関数名, default: ELU
            norm=config.goal_enc_use_layer_norm,  # GoalAEのlayer normの有無, default: True
            dist=config.goal_enc_dist_type,  # GoalAEの分布タイプ, default: onehot   
            std=config.goal_enc_std,  # GoalAEの標準偏差, default: 'none'
            outscale=config.goal_enc_outscale,  # GoalAEの出力スケール, default: 0.1
            unimix_ratio=config.goal_enc_unimix_ratio,  # GoalAEのunimix比率, default: 0.0
            name="GoalEncoder",
        )
        # 形状(config.goal_enc_num_codebook, config.goal_enc_num_categorical)のUniform Categorical分布を仮定，
        self.goal_enc_prior = [config.goal_enc_num_codebook, config.goal_enc_num_categorical]
        self.goal_dec = networks.MLP(
            inp_dim=config.goal_enc_num_codebook * config.goal_enc_num_categorical,  # goalAEの離散潜在の次元    
            shape=(config.dyn_deter,),  # TODO: 状態の次元, 今は決定論的状態を使用，Encに合わせる．
            layers=config.num_goal_dec_layers,  # GoalAEのMLP層数, default: 4
            units=config.goal_dec_hidden_units,  # GoalAEのMLPユニット数
            act=config.goal_dec_act_fn,  # GoalAEの活性化関数名, default: ELU
            norm=config.goal_dec_use_layer_norm,  # GoalAEのlayer normの有無
            dist=config.goal_dec_dist_type,  # GoalAEの分布タイプ, default: mse
            std=config.goal_dec_std,  # GoalAEの標準偏差, default: 'none'
            outscale=config.goal_dec_outscale,  # GoalAEの出力スケール, default: 0.1
            name="GoalDecoder",
        )
        
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        
        model_params = []
        goal_ae_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("goal_enc.") or name.startswith("goal_dec."):
                goal_ae_params.append(p)
            else:
                model_params.append(p)
                
        self.model_params = model_params
        self.goal_ae_params = goal_ae_params
        self._model_opt = tools.Optimizer(
            "model",
            model_params,
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

        # Goal AEの最適化設定はWMと同じ（公式実装と同様）
        self._goal_ae_opt = tools.Optimizer(
            "goal_ae",
            goal_ae_params,
            getattr(config, "goal_ae_lr", config.model_lr),      
            getattr(config, "goal_ae_opt_eps", config.opt_eps),
            getattr(config, "goal_ae_grad_clip", config.grad_clip),
            getattr(config, "goal_ae_weight_decay", config.weight_decay),
            opt=getattr(config, "goal_ae_opt", config.opt),
            use_amp=self._use_amp,
        )
        
        # Goal AE用のKL scaleクラス
        encdec_kl_cfg = dict(impl="mult", scale=0.0, target=10.0, min=1e-5, max=1.0)
        self.kl_scaler = tools.AutoAdapt(
            shape=(),
            impl=encdec_kl_cfg["impl"],
            scale=encdec_kl_cfg["scale"],
            target=encdec_kl_cfg["target"],
            min=encdec_kl_cfg["min"],
            max=encdec_kl_cfg["max"],
        )

        print(f"Optimizer model_opt has {sum(p.numel() for p in model_params)} variables.")
        print(f"Optimizer goal_ae_opt has {sum(p.numel() for p in goal_ae_params)} variables.")
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                
                # VTA returns additional kl_mask value
                if self._dynamics_type == 'vta':
                    mask_scale = getattr(self._config, 'vta_mask_scale', 1.0)
                    kl_loss, kl_value, dyn_loss, rep_loss, kl_mask = self.dynamics.kl_loss(
                        post, prior, kl_free, dyn_scale, rep_scale, mask_scale
                    )
                else:
                    kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                        post, prior, kl_free, dyn_scale, rep_scale
                    )
                    kl_mask = None
                
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                boundary_reg = 0.0
                boundary_kl = None
                if self._dynamics_type == 'vta':
                    rate = getattr(self._config, "vta_boundary_rate", 0.0)
                    scale = getattr(self._config, "vta_boundary_scale", 0.0)
                    if scale and rate > 0:
                        probs = torch.softmax(post["boundary_logit"], dim=-1)[..., 0]
                        target = torch.full_like(probs, float(rate))
                        eps = 1e-6
                        boundary_kl = probs * (
                            torch.log(probs + eps) - torch.log(target + eps)
                        ) + (1.0 - probs) * (
                            torch.log(1.0 - probs + eps) - torch.log(1.0 - target + eps)
                        )
                        boundary_reg = scale * boundary_kl
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                if boundary_kl is not None:
                    model_loss = model_loss + boundary_reg
            metrics = self._model_opt(torch.mean(model_loss), self.model_params)

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        if boundary_kl is not None:
            metrics["boundary_kl"] = to_np(torch.mean(boundary_kl))
        with torch.cuda.amp.autocast(self._use_amp):
            if self._dynamics_type == 'vta':
                # VTA-specific metrics
                metrics["prior_abs_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(prior, level="abs").entropy())
                )
                metrics["post_abs_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(post, level="abs").entropy())
                )
                metrics["prior_obs_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(prior, level="obs").entropy())
                )
                metrics["post_obs_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(post, level="obs").entropy())
                )
                # Boundary ratio (how often boundaries are detected)
                metrics["boundary_ratio"] = to_np(torch.mean(post["boundary"]))
                # Boundary KL (kl_mask) - measures how well prior boundary matches posterior
                if kl_mask is not None:
                    metrics["kl_mask"] = to_np(torch.mean(kl_mask))
                context = dict(
                    embed=embed,
                    feat=self.dynamics.get_feat(post),
                    deter_feat=self.dynamics.get_deter_obs_feat(post),  # TODO: dynamicsモデル（VTA）から決定状態のみ取得
                    kl=kl_value,
                    postent=self.dynamics.get_dist(post, level="obs").entropy(),
                )
            else:
                metrics["prior_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(prior).entropy())
                )
                metrics["post_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(post).entropy())
                )
                context = dict(
                    embed=embed,
                    feat=self.dynamics.get_feat(post),
                    deter_feat=self.dynamics.get_deter_feat(post),
                    kl=kl_value,
                    postent=self.dynamics.get_dist(post).entropy(),
                )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics
    
    def _train_goal_ae(self, cached_states, flow_grad_to_wm=False):
        """Goal AutoEncoderの訓練
        Args:
            cached_states: 状態表現のキャッシュ (batch_size, length, state_dim)
            flow_grad_to_wm: World Modelへの勾配伝播を許可するかどうか, 原著ではFalse (Sec.2.2の最後)
        Returns:
        """
        B, L, D = cached_states.shape
        # flatten to (B*L, D)
        org_states = cached_states.reshape(B * L, D)
        if not flow_grad_to_wm:
            org_states = org_states.detach()
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Encode
                enc_dist = self.goal_enc(org_states) 
                z_sample = enc_dist.sample()  # (B*L, num_codebook, num_categorical)
                z_sample = z_sample.reshape(B * L, -1)  # flatten to (B*L, num_codebook * num_categorical)
                
                # Decode
                dec_dist = self.goal_dec(z_sample)
                rec_states = dec_dist.mean() # (B*L, D)
                
                # Compute reconstruction loss & prior kl loss
                # rec_loss = torch.mean((rec_states - org_states) ** 2)
                rec_loss = torch.sum((rec_states - org_states) ** 2, dim=-1).mean()
                pred_probs = enc_dist.probs  # (B*L, num_codebook, num_categorical)
                prior_dist = torch.ones_like(pred_probs) / pred_probs.size(-1)
                kl_loss = torch.sum(
                    pred_probs * (torch.log(pred_probs + 1e-10) - torch.log(prior_dist + 1e-10)), dim=-1
                ).mean()
                
                # total loss
                # ★修正: 余分に掛けられていた固定scaleを外し、公式のAutoAdaptだけによるスケーリングを適用する
                kl_scaled_loss, kl_mets = self.kl_scaler(kl_loss)
                total_loss = rec_loss + kl_scaled_loss
            
            metrics = self._goal_ae_opt(torch.mean(total_loss), self.goal_ae_params)
        metrics["goal_ae_rec_loss"] = to_np(rec_loss)
        metrics["goal_ae_kl_loss"] = to_np(kl_loss)
        metrics["goal_ae_kl_scale"] = to_np(self.kl_scaler.scale()) # スケール監視用
        return metrics
    
    def decode_goal_ae_latent(self, goal_latent):
        """Goal AutoEncoderの潜在変数から状態を再構成する
        Args:
            goal_latent: Goal AutoEncoderの潜在変数 (batch_size, num_codebook, num_categorical) 
        Returns:
            再構成された状態 (batch_size, state_dim)
        """
        dec_dist = self.goal_dec(goal_latent)
        rec_states = dec_dist.mean()
        return rec_states
    
    def reconstruct_goal_ae(self, states):
        """Goal AutoEncoderによる状態の再構成
        Args:
            states: 再構成対象の状態 (batch_size, state_dim)
        Returns:
            再構成された状態 (batch_size, state_dim)
        """
        B = states.size(0)
        enc_dist = self.goal_enc(states)
        z_sample = enc_dist.sample()
        z_sample = z_sample.reshape(B, -1)  # flatten
        dec_dist = self.goal_dec(z_sample)
        rec_states = dec_dist.mean()
        return rec_states
    
    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        
        # Handle VTA imagination with boundary_mode parameter
        if self._dynamics_type == 'vta':
            prior = self.dynamics.imagine_with_action(
                data["action"][:6, 5:], init,
                boundary_mode=getattr(self._config, 'vta_imag_boundary', 'prior')
            )
        else:
            prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        # Ensure model and truth have the same sequence length
        min_len = min(model.shape[1], truth.shape[1])
        model = model[:, :min_len]
        truth = truth[:, :min_len]
        error = (model - truth + 1.0) / 2.0

        # For VTA, add boundary indicator row
        if self._dynamics_type == 'vta':
            # Get boundary values from states and prior
            # states boundary: (batch, time)
            # prior boundary: (batch, time) 
            obs_boundary = states.get("boundary", None)
            imag_boundary = prior.get("boundary", None)
            
            if obs_boundary is not None and imag_boundary is not None:
                # Combine observed and imagined boundaries
                # obs_boundary: (6, 5, 1), imag_boundary: (6, time, 1)
                all_boundary = torch.cat([obs_boundary[:, :5], imag_boundary], dim=1)  # (6, total_time, 1)
                # Truncate to match the same length as truth/model/error
                all_boundary = all_boundary[:, :min_len]
                
                # Create boundary visualization strip
                # Shape: (batch, time, height, width, channels)
                batch_size, seq_len = all_boundary.shape[:2]
                img_height, img_width = truth.shape[2:4]
                strip_height = max(4, img_height // 16)  # Height of the color strip
                
                # Create RGB strip: Red for READ (boundary=1), Blue for COPY (boundary=0)
                boundary_strip = torch.zeros(
                    batch_size, seq_len, strip_height, img_width, 3,
                    device=truth.device, dtype=truth.dtype
                )
                
                # Expand boundary to match strip dimensions
                boundary_expanded = all_boundary.squeeze(-1)  # (batch, time)
                
                for b in range(batch_size):
                    for t in range(seq_len):
                        m = boundary_expanded[b, t].item()
                        if m > 0.5:  # READ (boundary)
                            # Red color
                            boundary_strip[b, t, :, :, 0] = 1.0  # R
                            boundary_strip[b, t, :, :, 1] = 0.0  # G
                            boundary_strip[b, t, :, :, 2] = 0.0  # B
                        else:  # COPY (no boundary)
                            # Blue color
                            boundary_strip[b, t, :, :, 0] = 0.0  # R
                            boundary_strip[b, t, :, :, 1] = 0.0  # G
                            boundary_strip[b, t, :, :, 2] = 1.0  # B
                
                # Concatenate boundary strip below all three rows (truth, model, error)
                # Create a placeholder strip matching the 3-row height structure
                # We'll add it as a 4th row
                return torch.cat([truth, model, error, boundary_strip], 2)
        
        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        
        # Get feat_size from dynamics model
        dynamics_type = getattr(config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            feat_size_mgr = world_model.dynamics._abs_feat_size
            feat_size_wkr = world_model.dynamics._obs_feat_size
        elif config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            
        # Manager Policy (Manager actor and manager value)
        self.manager_actor = networks.MLP(
            inp_dim=feat_size_mgr,
            shape=(config.goal_enc_num_codebook, config.goal_enc_num_categorical),
            layers=config.manager_actor.get("num_layers", 3),
            units=config.manager_actor.get("hidden_units", 512),
            act=config.manager_actor.get("act_fn", "elu"),
            norm=config.manager_actor.get("use_layer_norm", True),
            dist=config.manager_actor.get("dist_type", "onehot"),
            std=config.manager_actor.get("std", "none"),
            outscale=config.manager_actor.get("outscale", 0.1),
            name="ManagerActor",
        )
        self.manager_rews = config.manager_rews
        self.worker_rews = config.worker_rews
        
        # Manager Value (now a ModuleDict)
        self.manager_values = nn.ModuleDict()
        for k, v in self.manager_rews.items():
            if v != 0.0:
                self.manager_values[k] = networks.MLP(
                    inp_dim=feat_size_mgr,
                    shape=(255,) if config.manager_critic.get("dist_type", "symlog_disc") == "symlog_disc" else (),
                    layers=config.manager_critic.get("num_layers", 4),
                    units=config.manager_critic.get("hidden_units", 512),
                    act=config.manager_critic.get("act_fn", "elu"),
                    norm=config.manager_critic.get("use_layer_norm", True),
                    dist=config.manager_critic.get("dist_type", "symlog_disc"),
                    outscale=config.manager_critic.get("outscale", 0.0),
                    name=f"ManagerValue_{k}",
                )
        
        # Worker Value (now a ModuleDict)
        # TODO: 今はGoalサイズは時間抽象の決定状態 
        goal_size = world_model.dynamics._abs_belief_size
        self.worker_actor = networks.MLP(
            inp_dim = feat_size_wkr + goal_size,
            shape = (config.num_actions,),
            layers=config.worker_actor.get("num_layers", 3),
            units=config.worker_actor.get("hidden_units", 512),
            act=config.worker_actor.get("act_fn", "elu"),
            norm=config.worker_actor.get("use_layer_norm", True),
            dist=config.worker_actor.get("dist_type", "normal"),
            std=config.worker_actor.get("std", "learned"),
            outscale=config.worker_actor.get("outscale", 0.1),
            name="WorkerActor",
        )

        self.worker_values = nn.ModuleDict()
        for k, v in self.worker_rews.items():
            if v != 0.0:
                self.worker_values[k] = networks.MLP(
                    inp_dim=feat_size_wkr + goal_size,
                    shape=(255,) if config.worker_critic.get("dist_type", "symlog_disc") == "symlog_disc" else (),
                    layers=config.worker_critic.get("num_layers", 4),
                    units=config.worker_critic.get("hidden_units", 512),
                    act=config.worker_critic.get("act_fn", "elu"),
                    norm=config.worker_critic.get("use_layer_norm", True),
                    dist=config.worker_critic.get("dist_type", "symlog_disc"),
                    outscale=config.worker_critic.get("outscale", 0.0),
                    name=f"WorkerValue_{k}",
                )

        if config.critic["slow_target"]:
            self._slow_values_wkr = nn.ModuleDict({k: copy.deepcopy(v) for k, v in self.worker_values.items()})
            self._slow_values_mgr = nn.ModuleDict({k: copy.deepcopy(v) for k, v in self.manager_values.items()})
            for p in self._slow_values_wkr.parameters():
                p.requires_grad = False
            for p in self._slow_values_mgr.parameters():
                p.requires_grad = False
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)

        self.wkr_actor_opt = tools.Optimizer(
            "worker_actor",
            self.worker_actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        self.mgr_actor_opt = tools.Optimizer(
            "manager_actor",
            self.manager_actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        
        print(f"Worker Actor: {self.worker_actor}")
        print(f"Manager Actor: {self.manager_actor}")
        print(f"Worker Actor Optimizer: {self.wkr_actor_opt}")
        print(f"Manager Actor Optimizer: {self.mgr_actor_opt}")
        
        # Manager Value optim.
        self.mgr_value_opt = tools.Optimizer(
            "manager_value",
            self.manager_values.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(f"Manager Values: {self.manager_values}")
        
        # Worker Value optim.
        self.wkr_value_opt = tools.Optimizer(
            "worker_value",
            self.worker_values.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(f"Worker Values: {self.worker_values}")

        if self._config.reward_EMA:
            # Worker 用 EMA（goal 報酬など）
            self.register_buffer(
                "wkr_ema_vals", torch.zeros((2,), device=self._config.device)
            )
            # Manager 用 EMA（extr + expl 報酬など）
            self.register_buffer(
                "mgr_ema_vals", torch.zeros((2,), device=self._config.device)
            )

        
        # ★ 公式 Director 準拠: Entropy AutoAdapt + Advantage Normalize
        self.wkr_actent = tools.AutoAdapt(
            shape=(), impl='mult', scale=3e-3, target=0.5,
            min=1e-5, max=1e2, vel=0.1, thres=0.1, inverse=True,
        )
        self.mgr_actent = tools.AutoAdapt(
            shape=(), impl='mult', scale=3e-3, target=0.5,
            min=1e-5, max=1e2, vel=0.1, thres=0.1, inverse=True,
        )
        
        self.wkr_advnorms = nn.ModuleDict()
        self.wkr_retnorms = nn.ModuleDict()
        for k, v in self.worker_rews.items():
            if v != 0.0:
                self.wkr_advnorms[k] = tools.Normalize(impl='mean_std', decay=0.99, max=1e8)
                self.wkr_retnorms[k] = tools.Normalize(impl='std', decay=0.999, max=1e2)
                
        self.mgr_advnorms = nn.ModuleDict()
        self.mgr_retnorms = nn.ModuleDict()
        for k, v in self.manager_rews.items():
            if v != 0.0:
                self.mgr_advnorms[k] = tools.Normalize(impl='mean_std', decay=0.99, max=1e8)
                self.mgr_retnorms[k] = tools.Normalize(impl='std', decay=0.999, max=1e2)
            
    def _cosine_max_similarity(self, pred, target):
        """"Cosine Max Similarity between pred and target.
        Args: 
            pred: (batch, dim)
            target: (batch, dim)
        Returns:
            cosine similarity: (batch,)
        """
        pred_norm = torch.linalg.norm(pred, dim=-1, keepdim=True) + 1e-12
        target_norm = torch.linalg.norm(target, dim=-1, keepdim=True) + 1e-12
        norm = torch.maximum(pred_norm, target_norm)
    
        cosine = torch.sum((pred / norm) * (target / norm), dim=-1)
        return cosine  # (batch,)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        # Step 1: Imagination - 共通の trajectory を生成
        import time
        s_time = time.time()
        with tools.RequiresGrad(self.manager_actor), tools.RequiresGrad(self.worker_actor):
            with torch.cuda.amp.autocast(self._use_amp):
                s_time = time.time()
                imag_full_feat, imag_obs_feat, imag_obs_state, imag_action, goal_states, goal_latents = self._imagine(
                    start, self._config.imag_horizon
                )
                e_time = time.time()
                # print(f"Imagination time: {e_time - s_time:.2f} seconds")
                
                # rewardは3種類ある
                # i) extrinsic reward (extr_reward): 環境からの報酬 -> world_modelで予測, managerが最大化
                # ii) exploration reward (expl_reward): 探索報酬, -> goal aeより算出, managerが最大化
                # iii) goal reward (goal_reward): goal達成報酬 -> workerが最大化
                
                s_time = time.time()
                # Extrinsic Rewardの計算
                extr_reward = objective(imag_full_feat, imag_obs_state, imag_action)[1:]
                # extr_reward: (time-1, batch, 1)
                
                # Exploration Rewardの計算
                deter_states = imag_obs_state['obs_belief'][1:]
                L, B, D = deter_states.shape
                flat_states = deter_states.reshape(L * B, D)
                rec_states = self._world_model.reconstruct_goal_ae(flat_states)
                rec_states = rec_states.reshape(L, B, D)
                expl_reward = torch.mean((rec_states - deter_states) ** 2, dim=-1, keepdim=True) 
                
                # Goal Rewardの計算 (Worker 用なので goal_states を detach)
                flatten_goal_states = goal_states[1:].detach().reshape(L*B, -1)
                goal_reward = self._cosine_max_similarity(
                    flat_states, 
                    flatten_goal_states,
                ).reshape(L, B, 1)
                
                # ★ 公式実装に合わせて trajectory を構築
                # 参照: director/embodied/agents/director/hierarchy.py train_jointly (L144-162)
                # cont は全タイムステップ（horizon+1）に対して定義
                if "cont" in self._world_model.heads:
                    discount = self._config.discount * self._world_model.heads["cont"](imag_full_feat).mean
                else:
                    discount = self._config.discount * torch.ones(
                        imag_full_feat.shape[0], imag_full_feat.shape[1], 1,
                        device=imag_full_feat.device, dtype=imag_full_feat.dtype)
                
                # 共通 trajectory
                # ★ goal_states と goal_latents を detach して Worker と Manager の勾配グラフを分離
                traj = {
                    'feat': imag_obs_feat,  # Worker用: obs_feat (544)
                    'goal': goal_states.detach(),  # Worker はこれを使う（Manager から切り離す）
                    'skill': goal_latents.detach(),  # 同様に detach
                    'action': imag_action,
                    'reward_extr': extr_reward,
                    'reward_expl': expl_reward,
                    'reward_goal': goal_reward,
                    'cont': discount,
                }
                
                # ★ Worker 用: split_traj を適用
                k = self._config.train_skill_duration
                
                wtraj = self._split_traj(traj, k)
                # ★ Manager 用: abstract_traj を適用
                mtraj = self._abstract_traj(traj, k)
                # Manager は goal_latents からの勾配が必要なので、抽象化した skill を直接設定
                T = goal_latents.shape[0]
                n_abstract = (T - 1) // k
                if n_abstract > 0:
                    # 抽象化: K ステップごとに最初の状態のみ使用
                    indices = [i * k for i in range(n_abstract + 1)]
                    mtraj['skill'] = goal_latents[indices]
                
                e_time = time.time()
                # print(f"Trajectory processing time: {e_time - s_time:.2f} seconds")
                
                # ========== Worker の訓練 ==========
                s_time = time.time()
                wkr_value_inp = torch.cat([wtraj['feat'], wtraj['goal']], dim=-1)
                
                # Worker の各報酬成分について Advantage を計算
                wkr_advs = []
                wkr_targets_dict = {}
                discount = self._config.discount
                wkr_weights = (torch.cumprod(wtraj['cont'], dim=0) / discount).detach()
                
                for key, scale in self.worker_rews.items():
                    if scale == 0.0:
                        continue
                    
                    wkr_reward = wtraj[f'reward_{key}']
                    with torch.no_grad():
                        wkr_value = self._slow_values_wkr[key](wkr_value_inp).mode()
                        
                    wkr_target = tools.lambda_return(
                        wkr_reward,
                        wkr_value[:-1],
                        wtraj['cont'][1:],
                        bootstrap=wkr_value[-1],
                        lambda_=self._config.discount_lambda,
                        axis=0,
                    )
                    
                    wkr_target_tensor = torch.stack(wkr_target, dim=1) if isinstance(wkr_target, (list, tuple)) else wkr_target
                    wkr_targets_dict[key] = wkr_target_tensor
                    
                    normed_target_wkr = self.wkr_retnorms[key](wkr_target_tensor)
                    normed_base_wkr = self.wkr_retnorms[key](wkr_value[:-1], update=False)
                    adv_wkr_k = self.wkr_advnorms[key](normed_target_wkr - normed_base_wkr)
                    wkr_advs.append(adv_wkr_k * scale)
                    
                adv_wkr = sum(wkr_advs)
                
                wkr_policy = self.worker_actor(wkr_value_inp.detach())
                
                # log_prob: (k+1, N*B) → [:-1] → (k, N*B) to match adv_wkr
                wkr_logprob = wkr_policy.log_prob(wtraj['action'].detach())[:-1].unsqueeze(-1)
                
                # ★ Worker はconfigに従う（Atari=reinforce, DMC=dynamics/backprop）
                if self._config.imag_gradient == 'reinforce':
                    wkr_actor_target = wkr_logprob * adv_wkr.detach()
                elif self._config.imag_gradient == 'dynamics':
                    wkr_actor_target = adv_wkr
                else:
                    raise NotImplementedError(self._config.imag_gradient)

                wkr_actor_loss = -wkr_weights[:-1] * wkr_actor_target
                # ★ 公式 Director 準拠: entropy 正規化 (連続/離散で分岐)
                if self._config.imag_gradient == 'dynamics':
                    import math
                    base_dist = wkr_policy._dist.base_dist  # Normal per-dim
                    wkr_ent = base_dist.entropy()[:-1]  # (T, B, act_dim)
                    min_std = self.worker_actor._min_std
                    max_std = self.worker_actor._max_std
                    lo = 0.5 * math.log(2 * math.pi * math.e * min_std ** 2)
                    hi = 0.5 * math.log(2 * math.pi * math.e * max_std ** 2)
                    wkr_ent_normed = (wkr_ent - lo) / (hi - lo)  # per-dim [0, 1]
                    wkr_ent_loss, wkr_ent_mets = self.wkr_actent(wkr_ent_normed)
                    wkr_ent_loss = wkr_ent_loss.sum(-1)  # 次元ごとに合計
                else:
                    wkr_ent = wkr_policy.entropy()[:-1]
                    wkr_maxent = torch.log(torch.tensor(float(self._config.num_actions), device=wkr_ent.device))
                    wkr_ent_normed = wkr_ent / wkr_maxent
                    wkr_ent_loss, wkr_ent_mets = self.wkr_actent(wkr_ent_normed)

                wkr_actor_loss = (-wkr_actor_target + wkr_ent_loss.unsqueeze(-1)) * wkr_weights[:-1]
                wkr_actor_loss = torch.mean(wkr_actor_loss)
                e_time = time.time()
                # print(f"Worker policy loss time: {e_time - s_time:.3f} sec")
                
                # ========== Manager の訓練 ==========
                s_time = time.time()
                mgr_value_inp = mtraj['feat']
                
                # Manager の各報酬成分について Advantage を計算
                mgr_advs = []
                mgr_targets_dict = {}
                mgr_weights = (torch.cumprod(mtraj['cont'], dim=0) / discount).detach()
                
                for key, scale in self.manager_rews.items():
                    if scale == 0.0:
                        continue
                        
                    mgr_reward = mtraj[f'reward_{key}']
                    with torch.no_grad():
                        mgr_value = self._slow_values_mgr[key](mgr_value_inp).mode()
                        
                    mgr_target = tools.lambda_return(
                        mgr_reward,
                        mgr_value[:-1],
                        mtraj['cont'][1:],
                        bootstrap=mgr_value[-1],
                        lambda_=self._config.discount_lambda,
                        axis=0,
                    )
                    
                    mgr_target_tensor = torch.stack(mgr_target, dim=1) if isinstance(mgr_target, (list, tuple)) else mgr_target
                    mgr_targets_dict[key] = mgr_target_tensor
                    
                    normed_target_mgr = self.mgr_retnorms[key](mgr_target_tensor)
                    normed_base_mgr = self.mgr_retnorms[key](mgr_value[:-1], update=False)
                    adv_mgr_k = self.mgr_advnorms[key](normed_target_mgr - normed_base_mgr)
                    mgr_advs.append(adv_mgr_k * scale)
                    
                adv_mgr = sum(mgr_advs)
                    
                mgr_policy = self.manager_actor(mgr_value_inp.detach())
                
                # Reshape skill for log_prob calculation
                skill_flat = mtraj['skill']
                L, B = skill_flat.shape[:2]
                skill_reshaped = skill_flat.reshape(L, B, self._config.goal_enc_num_codebook, self._config.goal_enc_num_categorical)
                
                # log_prob: (L, B, 8) → sum(-1) → (L, B). [:-1] to match adv_mgr
                mgr_logprob = mgr_policy.log_prob(skill_reshaped.detach()).sum(-1, keepdim=True)[:-1]

                # ★ Manager は常に reinforce（公式準拠: 離散ポリシー）
                mgr_actor_target = mgr_logprob * adv_mgr.detach()

                mgr_actor_loss = -mgr_weights[:-1] * mgr_actor_target
                
                # ★ 公式 Director 準拠: entropy を [0,1] に正規化し AutoAdapt
                mgr_ent = mgr_policy.entropy()[:-1]  # (n_abstract+1-1, B, num_codebook)
                mgr_maxent = torch.log(torch.tensor(float(self._config.goal_enc_num_categorical), device=mgr_ent.device))
                mgr_ent_normed = mgr_ent / mgr_maxent
                mgr_ent_loss, mgr_ent_mets = self.mgr_actent(mgr_ent_normed)
                mgr_ent_loss = mgr_ent_loss.sum(-1, keepdim=True)

                # ★ 公式準拠: (REINFORCE_loss + entropy_loss) * weight
                mgr_actor_loss = (-mgr_actor_target + mgr_ent_loss) * mgr_weights[:-1]
                mgr_actor_loss = torch.mean(mgr_actor_loss)
                
                # エントロピー計算（メトリクス用）- 既存の policy を使用
                manager_ent = mgr_policy.entropy()
                worker_ent = wkr_policy.entropy()
                e_time = time.time()
                # print(f"Manager policy loss time: {e_time - s_time:.3f} sec")
        e_time = time.time()
        # print(f"Imagination and policy loss time: {e_time - s_time:.3f} sec")
        
        s_time = time.time()
        # ========== Value の更新 ==========
        with tools.RequiresGrad(self.worker_values), tools.RequiresGrad(self.manager_values):
            with torch.cuda.amp.autocast(self._use_amp):
                # Worker value loss
                wkr_value_losses = []
                wkr_values_for_loss = {}
                for key, target_tensor in wkr_targets_dict.items():
                    val_k = self.worker_values[key](wkr_value_inp[:-1].detach())
                    wkr_values_for_loss[key] = val_k
                    loss_k = -1 * val_k.log_prob(target_tensor.detach())
                    loss_k = torch.mean(wkr_weights[:-1] * loss_k[:, :, None])
                    wkr_value_losses.append(loss_k)
                wkr_value_loss = sum(wkr_value_losses)
                
                # Manager value loss
                mgr_value_losses = []
                mgr_values_for_loss = {}
                for key, target_tensor in mgr_targets_dict.items():
                    val_k = self.manager_values[key](mgr_value_inp[:-1].detach())
                    mgr_values_for_loss[key] = val_k
                    loss_k = -1 * val_k.log_prob(target_tensor.detach())
                    loss_k = torch.mean(mgr_weights[:-1] * loss_k[:, :, None])
                    mgr_value_losses.append(loss_k)
                mgr_value_loss = sum(mgr_value_losses)
        e_time = time.time()
        # print(f"Value loss time: {e_time - s_time:.3f} sec")
        
        s_time = time.time()
        # Metrics
        for key, val_k in wkr_values_for_loss.items():
            metrics.update(tools.tensorstats(val_k.mode(), f"worker_value_{key}"))
        for key, val_k in mgr_values_for_loss.items():
            metrics.update(tools.tensorstats(val_k.mode(), f"manager_value_{key}"))
        metrics.update(tools.tensorstats(extr_reward, "imag_reward"))
        metrics.update(tools.tensorstats(expl_reward, "imag_expl_reward"))
        metrics.update(tools.tensorstats(goal_reward, "imag_goal_reward"))
        metrics["EMA_005_wkr"] = to_np(self.wkr_ema_vals[0])
        metrics["EMA_095_wkr"] = to_np(self.wkr_ema_vals[1])
        metrics["EMA_005_mgr"] = to_np(self.mgr_ema_vals[0])
        metrics["EMA_095_mgr"] = to_np(self.mgr_ema_vals[1])
        
        # ★ AutoAdapt / Normalize 内部状態の監視
        metrics["wkr_actent_scale"] = to_np(self.wkr_actent.scale())
        metrics["mgr_actent_scale"] = to_np(self.mgr_actent.scale())
        metrics["wkr_ent_normed_mean"] = to_np(wkr_ent_mets["mean"])
        metrics["mgr_ent_normed_mean"] = to_np(mgr_ent_mets["mean"])
        
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_wkr_entropy"] = to_np(torch.mean(worker_ent))
        metrics["actor_mgr_entropy"] = to_np(torch.mean(manager_ent))
        metrics["wkr_actor_loss"] = to_np(wkr_actor_loss)
        metrics["mgr_actor_loss"] = to_np(mgr_actor_loss)
        metrics["wkr_value_loss"] = to_np(wkr_value_loss)
        metrics["mgr_value_loss"] = to_np(mgr_value_loss)

        # Optimizer 更新
        with tools.RequiresGrad(self.manager_actor):
            metrics.update(self.mgr_actor_opt(mgr_actor_loss, self.manager_actor.parameters()))

        with tools.RequiresGrad(self.worker_actor):
            metrics.update(self.wkr_actor_opt(wkr_actor_loss, self.worker_actor.parameters()))

        with tools.RequiresGrad(self.manager_values):
            metrics.update(self.mgr_value_opt(mgr_value_loss, self.manager_values.parameters()))

        with tools.RequiresGrad(self.worker_values):
            metrics.update(self.wkr_value_opt(wkr_value_loss, self.worker_values.parameters()))
        
        # weights は wkr_weights を返す（既に正しく計算済み）
        e_time = time.time()
        return imag_obs_feat, imag_obs_state, imag_action, wkr_weights, metrics

    def _imagine(self, start, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        
        # Check if using VTA
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            return self._imagine_vta_full(start, horizon, dynamics)
        else:
            return self._imagine_rssm(start, horizon, dynamics)
    
    def _imagine_rssm(self, start, horizon, dynamics):
        """RSSM imagination with Manager & Worker.
        
        ★ 公式 imagine_carry (agent.py L227-255) に準拠:
        T = horizon + 1 のタイムステップを生成する。
        最後のステップはbootstrap用（遷移なし）。
        
        Returns:
            feats: (horizon+1, B, feat_size)
            states: dict of (horizon+1, B, ...)
            actions: (horizon+1, B, act_dim)
            goal_states: (horizon+1, B, deter_dim)
            goal_latents: (horizon+1, B, skill_dim)
        """
        B = list(start.values())[0].shape[0]
        k = self._config.train_skill_duration
        
        # Collect trajectory
        all_states = [start]
        all_feats = []
        all_actions = []
        all_goal_states = []
        all_goal_latents = []
        
        state = start
        prev_goal_state = None
        prev_goal_latent = None
        
        for t in range(horizon + 1):
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            
            # Manager: generate goal every k steps
            if t % k == 0:
                goal_dist = self.manager_actor(inp)
                goal_sample = goal_dist.sample().reshape(B, -1)
                goal_state = self._world_model.decode_goal_ae_latent(goal_sample)
                goal_latent = goal_sample
            else:
                goal_state = prev_goal_state
                goal_latent = prev_goal_latent
            
            # Worker: generate action
            worker_inp = torch.cat([inp, goal_state.detach()], dim=-1)
            action_dist = self.worker_actor(worker_inp)
            # ★ dynamics/backprop モードでは rsample() で再パラメータ化勾配を有効化
            # REINFORCE モード（離散行動）では sample() を使用（rsample非対応）
            if self._config.imag_gradient == 'dynamics' and hasattr(action_dist, 'rsample'):
                action = action_dist.rsample()
            else:
                action = action_dist.sample()
            
            # Collect
            all_feats.append(feat)
            all_actions.append(action)
            all_goal_states.append(goal_state)
            all_goal_latents.append(goal_latent)
            
            # Transition (except for the last step - bootstrap only)
            if t < horizon:
                state = dynamics.img_step(state, action)
                all_states.append(state)
            
            prev_goal_state = goal_state
            prev_goal_latent = goal_latent
        
        # Stack tensors: all have horizon+1 elements
        feats = torch.stack(all_feats, dim=0)
        actions = torch.stack(all_actions, dim=0)
        goal_states = torch.stack(all_goal_states, dim=0)
        goal_latents = torch.stack(all_goal_latents, dim=0)
        
        # Stack state dicts: horizon+1 elements [start, s1, ..., s_horizon]
        states = {}
        for key in all_states[0].keys():
            states[key] = torch.stack([s[key] for s in all_states], dim=0)
        
        return feats, feats, states, actions, goal_states, goal_latents
    
    def _imagine_vta_full(self, obs_start, horizon, dynamics):
        """VTA imagination with full step-by-step transitions.
        ★ 公式 imagine_carry (agent.py L227-255) に準拠:
        T = horizon + 1 のタイムステップを生成する。
        最後のステップはbootstrap用（遷移なし）。
        
        Returns:
            full_feats: (horizon+1, B, abs_feat+obs_feat) - for reward/cont heads
            obs_feats: (horizon+1, B, obs_feat_size) - for Worker actor/value
            obs_states: dict of (horizon+1, B, ...)
            actions: (horizon+1, B, act_dim)
            goal_states: (horizon+1, B, deter_dim)
            goal_latents: (horizon+1, B, skill_dim)
        """
        B = list(obs_start.values())[0].shape[0]
        k = self._config.train_skill_duration # TODO: VTAではkは可変
        
        # Collect trajectory
        all_obs_states = [obs_start]
        all_full_feats = []
        all_obs_feats = []
        all_actions = []
        all_goal_states = []
        all_goal_latents = []
        
        state = obs_start
        prev_goal_state = None
        prev_goal_latent = None
        
        for t in range(horizon + 1):
            full_feat = dynamics.get_feat(state)         # abs+obs (1088) → reward/cont head 用
            abs_feat = dynamics._get_abs_feat(state)      # abs only (544) → Manager 用
            obs_feat = dynamics._get_obs_feat(state)      # obs only (544) → Worker 用
            read_mask = state["boundary"]  # VTAの境界マスク(read), 1なら境界, (B, )
            
            # Manager: generate goal if boundary detected.
            generate_goal = read_mask.any() 
            if generate_goal:
                goal_dist = self.manager_actor(abs_feat.detach())
                goal_sample = goal_dist.sample().reshape(B, -1)
                new_goal_state = self._world_model.decode_goal_ae_latent(goal_sample)
                
                if prev_goal_state is not None:
                    gen_mask = read_mask.float().unsqueeze(-1)  # (B, 1)
                    goal_state  = gen_mask * new_goal_state  + (1 - gen_mask) * prev_goal_state
                    goal_latent = gen_mask * goal_sample + (1 - gen_mask) * prev_goal_latent
                else:
                    # t=0: 前のゴールがないので全サンプル新ゴール
                    goal_state = new_goal_state
                    goal_latent = goal_sample
            else:
                goal_state = prev_goal_state
                goal_latent = prev_goal_latent
            
            # Worker: generate action
            worker_inp = torch.cat([obs_feat.detach(), goal_state.detach()], dim=-1)
            action_dist = self.worker_actor(worker_inp)
            # ★ dynamics/backprop モードでは rsample() で再パラメータ化勾配を有効化
            # REINFORCE モード（離散行動）では sample() を使用（rsample非対応）
            if self._config.imag_gradient == 'dynamics' and hasattr(action_dist, 'rsample'):
                action = action_dist.rsample()
            else:
                action = action_dist.sample()
                
            # Collect
            all_full_feats.append(full_feat)
            all_obs_feats.append(obs_feat)
            all_actions.append(action)
            all_goal_states.append(goal_state)
            all_goal_latents.append(goal_latent)
            
            # Transition (except for the last step - bootstrap only)
            if t < horizon:
                state = dynamics.img_step(state, goal_state, action)
                # obs_state = dynamics._get_obs_state(state)  # VTAの観測抽象状態
                # all_obs_states.append(obs_state)
                all_obs_states.append(state)
            
        # Stack tensors: all have horizon+1 elements
        full_feats = torch.stack(all_full_feats, dim=0)
        obs_feats = torch.stack(all_obs_feats, dim=0)
        actions = torch.stack(all_actions, dim=0)
        goal_states = torch.stack(all_goal_states, dim=0)
        goal_latents = torch.stack(all_goal_latents, dim=0)           
        
        # Stack state dicts: horizon+1 elements [start, s1, ..., s_horizon]
        obs_states = {}
        for key in all_obs_states[0].keys():
            obs_states[key] = torch.stack([s[key] for s in all_obs_states], dim=0)
        
        return full_feats, obs_feats, obs_states, actions, goal_states, goal_latents
            
            

    # def _imagine_vta(self, start, horizon, dynamics):
    #     """VTA imagination with jumpy or full modes."""
    #     imag_type = getattr(self._config, 'vta_imag_type', 'full')
    #     boundary_mode = getattr(self._config, 'vta_imag_boundary', 'prior')
        
    #     if imag_type == 'jumpy':
    #         raise NotImplementedError("Jumpy imagination is not implemented yet.")
    #         # Jumpy imagination: each step is an abstract state transition
    #         def step(prev, _):
    #             state, _, _ = prev
    #             feat = dynamics.get_feat(state)
    #             inp = feat.detach()
    #             action = policy(inp).sample()
    #             succ = dynamics.jumpy_img_step(state, action)
    #             return succ, feat, action
    #     else:
    #         # Full imagination: normal timestep-level transitions
    #         def step(prev, _):
    #             state, _, _ = prev
    #             feat = dynamics.get_feat(state)
    #             inp = feat.detach()
    #             action = policy(inp).sample()
    #             succ = dynamics.img_step(state, action, boundary_mode=boundary_mode)
    #             return succ, feat, action

    #     succ, feats, actions = tools.static_scan(
    #         step, [torch.arange(horizon)], (start, None, None)
    #     )
    #     states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
    #     return feats, states, actions


    


    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                
                for key in self._slow_values_wkr.keys():
                    slow_wkr_value = self._slow_values_wkr[key]
                    wkr_value = self.worker_values[key]
                    for s, d in zip(wkr_value.parameters(), slow_wkr_value.parameters()):
                        d.data = mix * s.data + (1 - mix) * d.data
                        
                for key in self._slow_values_mgr.keys():
                    slow_mgr_value = self._slow_values_mgr[key]
                    mgr_value = self.manager_values[key]
                    for s, d in zip(mgr_value.parameters(), slow_mgr_value.parameters()):
                        d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
    
    # ★ 公式実装に合わせて split_traj を追加
    # 参照: director/embodied/agents/director/hierarchy.py L410-429
    def _split_traj(self, traj, k):
        """Worker 用: trajectory を K ステップのセグメントに分割
        
        公式実装の動作:
        - (1 2 3 4 5 6 7 8 9 10...) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10)...)
        - 各セグメントで同じ goal を追うように変換
        - Worker は各セグメント内で goal に到達することを学習
        
        Args:
            traj: trajectory dict with keys like 'feat', 'action', 'goal', 'reward_*', etc.
                  shapes: (time, batch, ...)
            k: skill_duration (segment length)
        Returns:
            split trajectory for worker training
        """
        new_traj = {}
        T = traj['feat'].shape[0]  # time dimension
        B = traj['feat'].shape[1]  # batch dimension
        
        # 公式実装: len(traj['action']) % k == 1 を前提
        # つまり T = N*k + 1 (例: 17 = 2*8 + 1) の形式
        # ★ 公式: (1 2 3 4 5 6 7 8 9) -> ((1 2 3 4) (4 5 6 7) (7 8 9))
        # stride = k-1 でオーバーラップするセグメントを作成
        
        n_segments = (T - 1) // k  # 完全なセグメント数
        if n_segments == 0:
            # k より短い場合はそのまま返す
            return traj
        
        for key, val in traj.items():
            if 'reward' in key:
                # reward は先頭に 0 を追加してから reshape
                # 公式: val = tf.concat([0 * val[:1], val], 0) if 'reward' in key else val
                val_padded = torch.cat([torch.zeros_like(val[:1]), val], dim=0)
                # ★ 公式: オーバーラップするセグメント化
                # stride = k-1 を使用して、境界要素が重複するセグメントを作成
                segments = []
                for i in range(n_segments):
                    # オーバーラップ: i=0: [0,k+1), i=1: [k-1, 2k), i=2: [2k-2, 3k-1)...
                    # 簡易版: 各セグメントは k+1 要素（reward 用にパディング済み）
                    start_idx = i * k
                    end_idx = start_idx + k + 1
                    seg = val_padded[start_idx:end_idx]
                    segments.append(seg)
                # Stack and reshape: (n_seg, k+1, B, ...) -> (k+1, n_seg*B, ...)
                stacked = torch.stack(segments, dim=0)  # (n_seg, k+1, B, ...)
                stacked = stacked.transpose(0, 1)  # (k+1, n_seg, B, ...)
                stacked = stacked.reshape(stacked.shape[0], -1, *stacked.shape[3:])  # (k+1, n_seg*B, ...)
                new_traj[key] = stacked[1:]  # remove padded zero: (k, n_seg*B, ...)
            else:
                # state/feat 等は直接セグメント化
                segments = []
                for i in range(n_segments):
                    start_idx = i * k
                    end_idx = start_idx + k + 1  # +1 for bootstrap state
                    seg = val[start_idx:end_idx]
                    segments.append(seg)
                stacked = torch.stack(segments, dim=0)  # (n_seg, k+1, B, ...)
                stacked = stacked.transpose(0, 1)  # (k+1, n_seg, B, ...)
                stacked = stacked.reshape(stacked.shape[0], -1, *stacked.shape[3:])  # (k+1, n_seg*B, ...)
                new_traj[key] = stacked  # (k+1, n_seg*B, ...)
        
        # ★ goal の bootstrap 処理
        # 公式: traj['goal'] = tf.concat([traj['goal'][:-1], traj['goal'][:1]], 0)
        # 各セグメントの最後の goal を最初の goal で置き換え（同じ goal を追い続けるため）
        if 'goal' in new_traj:
            goal = new_traj['goal']
            new_traj['goal'] = torch.cat([goal[:-1], goal[:1]], dim=0)
        
        return new_traj

    # ★ 公式実装に合わせて abstract_traj を追加
    # 参照: director/embodied/agents/director/hierarchy.py L431-446
    def _abstract_traj(self, traj, k):
        """Manager 用: K ステップを抽象化して 1 ステップに集約
        
        公式実装の動作:
        - action を skill に置き換え
        - reward は K ステップの加重平均
        - cont は K ステップの積
        - state は K ステップごとに最初の状態のみ
        
        Args:
            traj: trajectory dict
            k: skill_duration
        Returns:
            abstracted trajectory for manager training
        """
        new_traj = {}
        T = traj['feat'].shape[0]
        n_abstract = (T - 1) // k  # 抽象化後のステップ数
        
        if n_abstract == 0:
            return traj
        
        # ★ action を skill に置き換え
        # 公式: traj['action'] = traj.pop('skill')
        if 'skill' in traj:
            new_traj['action'] = traj['skill']
        
        for key, val in traj.items():
            if key == 'skill':
                continue  # already handled
            elif 'reward' in key:
                # ★ 公式実装: reward は cont による重み付け平均
                # 公式: weights = tf.math.cumprod(reshape(traj['cont'][:-1]), 1)
                #       traj[key] = (reshape(value) * weights).mean(1)
                segments = []
                for i in range(n_abstract):
                    start_idx = i * k
                    end_idx = start_idx + k
                    seg = val[start_idx:end_idx]  # (k, B, 1)
                    # cont_seg は該当区間の cont (kステップ分)
                    cont_seg = traj['cont'][start_idx:end_idx]
                    weights = torch.cumprod(cont_seg, dim=0)  # (k, B, 1)
                    seg_mean = (seg * weights).mean(dim=0, keepdim=True)  # (1, B, 1)
                    segments.append(seg_mean)
                new_traj[key] = torch.cat(segments, dim=0)  # (n_abstract, B, 1) ★公式準拠: val[-1:]は追加しない
            elif key == 'cont':
                # cont は K ステップの積
                # 公式: traj[key] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)
                segments = []
                for i in range(n_abstract):
                    start_idx = i * k + 1  # skip first
                    end_idx = start_idx + k
                    if end_idx <= T:
                        seg = val[start_idx:end_idx]
                        seg_prod = seg.prod(dim=0, keepdim=True)
                    else:
                        seg_prod = val[-1:]
                    segments.append(seg_prod)
                new_traj[key] = torch.cat([val[:1]] + segments, dim=0)
            else:
                # state/feat 等は K ステップごとに最初の状態のみ
                # 公式: traj[key] = tf.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
                segments = []
                for i in range(n_abstract):
                    start_idx = i * k
                    segments.append(val[start_idx:start_idx+1])
                new_traj[key] = torch.cat(segments + [val[-1:]], dim=0)  # (n_abstract+1, B, ...)
        
        return new_traj
            
    def take_action(self, feat, director_carry, training=False, latent=None):
        """現在の状態から行動を選択する.
        Args:
            feat: 現在の特徴量表現, (time, dim_deter + dim_stoch)
            director_carry: Directorの内部状態 (辞書型)
            latent: VTA状態辞書でありboundary情報を含む
        Returns:
            action: 選択された行動
            action_logprob: 選択された行動の対数確率
        """
        is_boundary = latent["boundary"]  # (batch, 1), 0.0 or 1.0

        # エピソード最初（goal が None）の場合は強制的に境界扱い
        if director_carry["goal"] is None:
            is_boundary = torch.ones_like(is_boundary)
        
        # バッチ内に1つでも境界があれば Manager を実行
        use_manager = is_boundary.any()
        # TODO: 固定間隔ではなくVTAによる境界検出に基づいて Manager を使用するように変更
        # skill_duration = self._config.train_skill_duration if training else self._config.eval_skill_duration
        # use_manager = director_carry["step"] % skill_duration == 0
        
        if use_manager:
            # Manager: 抽象状態 (abs_feat) からゴール生成
            abs_feat = self._world_model.dynamics._get_abs_feat(latent)
            L = abs_feat.size(0)
            goal_dist = self.manager_actor(abs_feat)
            new_goal_latent = goal_dist.sample()
            new_goal_sample = new_goal_latent.reshape(L, -1)
            new_goal_state = self._world_model.decode_goal_ae_latent(new_goal_sample)
            
            if director_carry["goal"] is not None:
                # mask演算: boundary=1 → 新ゴール, boundary=0 → 前ゴール維持
                goal_state  = is_boundary * new_goal_state  + (1 - is_boundary) * director_carry["goal"]
                goal_latent = is_boundary * new_goal_sample + (1 - is_boundary) * director_carry["skill"]
            else:
                goal_state = new_goal_state
                goal_latent = new_goal_sample
        else:
            # バッチ全体で境界なし → Manager をスキップ
            goal_state = director_carry["goal"]
            goal_latent = director_carry["skill"]
        
        director_carry["step"] += 1
        director_carry["goal"] = goal_state
        director_carry["skill"] = goal_latent

        # worker policyを用いて行動を推定.
        # worker_inp = torch.cat([feat, goal_state.detach()], dim=-1)
        # worker_dist = self.worker_actor(worker_inp)
        # action = worker_dist.sample()
        # action_logprob = worker_dist.log_prob(action)
        # return action, action_logprob, director_carry
        # Worker: 観測レベルの特徴 (obs_feat) + goal で行動生成
        obs_feat = self._world_model.dynamics._get_obs_feat(latent)
        worker_inp = torch.cat([obs_feat, goal_state.detach()], dim=-1)
        worker_dist = self.worker_actor(worker_inp)
        action = worker_dist.sample()
        action_logprob = worker_dist.log_prob(action)
        return action, action_logprob, director_carry
        
