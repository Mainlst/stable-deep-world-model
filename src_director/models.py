import copy
import torch
from torch import nn

from . import networks
from . import tools
from . import vta as vta_module

to_np = lambda x: x.detach().cpu().numpy()

import numpy as np

torch.autograd.set_detect_anomaly(True)

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    @torch.no_grad()
    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


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
                metrics = {}
                total_loss = rec_loss + self.kl_scaler(kl_loss)[0] * self._config.goal_ae_kl_scale
            
            metrics = self._goal_ae_opt(torch.mean(total_loss), self.goal_ae_params)
        metrics["goal_ae_rec_loss"] = to_np(rec_loss)
        metrics["goal_ae_kl_loss"] = to_np(kl_loss)
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
            feat_size = world_model.dynamics.feat_size
        elif config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            
        # Manager Policy (Manager actor and manager value)
        self.manager_actor = networks.MLP(
            inp_dim=feat_size,
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
        self.manager_value = networks.MLP(
            inp_dim=feat_size,
            shape=(255,) if config.manager_critic.get("dist_type", "symlog_disc") == "symlog_disc" else (),
            layers=config.manager_critic.get("num_layers", 4),
            units=config.manager_critic.get("hidden_units", 512),
            act=config.manager_critic.get("act_fn", "elu"),
            norm=config.manager_critic.get("use_layer_norm", True),
            dist=config.manager_critic.get("dist_type", "symlog_disc"),
            outscale=config.manager_critic.get("outscale", 0.0),
            name="ManagerValue",
        )
        
        self.manager_rews = config.manager_rews
        self.worker_rews = config.worker_rews
        
        # Worker Policy (Worker actor and worker value)
        # TODO: 今はGoal Encoderは決定論的状態(deter)を再構成しているので，goal_size = config.dyn_deter   
        goal_size = config.dyn_deter
        self.worker_actor = networks.MLP(
            inp_dim = feat_size + goal_size,
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

        self.worker_value = networks.MLP(
            inp_dim = feat_size + goal_size,
            shape=(255,) if config.worker_critic.get("dist_type", "symlog_disc") == "symlog_disc" else (),
            layers=config.worker_critic.get("num_layers", 4),
            units=config.worker_critic.get("hidden_units", 512),
            act=config.worker_critic.get("act_fn", "elu"),
            norm=config.worker_critic.get("use_layer_norm", True),
            dist=config.worker_critic.get("dist_type", "symlog_disc"),
            outscale=config.worker_critic.get("outscale", 0.0),
            name="WorkerValue",
        )
        
        if config.critic["slow_target"]:
            self._slow_value_wkr = copy.deepcopy(self.worker_value)
            self._slow_value_mgr = copy.deepcopy(self.manager_value)
            for p in self._slow_value_wkr.parameters():
                p.requires_grad = False
            for p in self._slow_value_mgr.parameters():
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
            self.manager_value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(f"Manager Value: {self.manager_value}")
        
        # Worker Value optim.
        self.wkr_value_opt = tools.Optimizer(
            "worker_value",
            self.worker_value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(f"Worker Value: {self.worker_value}")

        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            # self.register_buffer(
            #     "ema_vals", torch.zeros((2,), device=self._config.device)
            # )
            # self.reward_ema = RewardEMA(device=self._config.device)

            # 変更箇所
            # Worker 用 EMA（goal 報酬）
            self.register_buffer(
                "wkr_ema_vals", torch.zeros((2,), device=self._config.device)
            )
            # Manager 用 EMA（extr + expl 報酬）
            self.register_buffer(
                "mgr_ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)
        
        # ★ 公式 Director 準拠: Entropy AutoAdapt + Advantage Normalize
        # 公式: actent: {impl: mult, scale: 3e-3, target: 0.5, min: 1e-5, max: 1e2}
        # inverse=True: entropy を最大化する方向に損失を生成
        self.wkr_actent = tools.AutoAdapt(
            shape=(), impl='mult', scale=3e-3, target=0.5,
            min=1e-5, max=1e2, vel=0.1, thres=0.1, inverse=True,
        )
        self.mgr_actent = tools.AutoAdapt(
            shape=(), impl='mult', scale=3e-3, target=0.5,
            min=1e-5, max=1e2, vel=0.1, thres=0.1, inverse=True,
        )
        # 公式: advnorm: {impl: mean_std, decay: 0.99, max: 1e8}
        self.wkr_advnorm = tools.Normalize(impl='mean_std', decay=0.99, max=1e8)
        self.mgr_advnorm = tools.Normalize(impl='mean_std', decay=0.99, max=1e8)
        # 公式: retnorm: {impl: std, decay: 0.999, max: 1e2}
        self.wkr_retnorm = tools.Normalize(impl='std', decay=0.999, max=1e2)
        self.mgr_retnorm = tools.Normalize(impl='std', decay=0.999, max=1e2)
            
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
                imag_feat, imag_state, imag_action, goal_states, goal_latents = self._imagine(
                    start, self._config.imag_horizon
                )
                
                # rewardは3種類ある
                # i) extrinsic reward (extr_reward): 環境からの報酬 -> world_modelで予測, managerが最大化
                # ii) exploration reward (expl_reward): 探索報酬, -> goal aeより算出, managerが最大化
                # iii) goal reward (goal_reward): goal達成報酬 -> workerが最大化
                
                # Extrinsic Rewardの計算
                extr_reward = objective(imag_feat, imag_state, imag_action)[1:]
                # extr_reward: (time-1, batch, 1)
                
                # Exploration Rewardの計算
                deter_states = imag_state['deter'][1:]
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
                discount = self._config.discount * torch.ones(
                    imag_feat.shape[0], imag_feat.shape[1], 1,
                    device=imag_feat.device, dtype=imag_feat.dtype)
                
                # 共通 trajectory
                # ★ goal_states と goal_latents を detach して Worker と Manager の勾配グラフを分離
                traj = {
                    'feat': imag_feat,
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
                
                # ========== Worker の訓練 ==========
                # Worker の報酬: goal reward のみ
                wkr_reward = (self.worker_rews["extr"] * wtraj['reward_extr'] + 
                              self.worker_rews["expl"] * wtraj['reward_expl'] + 
                              self.worker_rews["goal"] * wtraj['reward_goal'])
                
                # Worker の value 入力: [feat, goal]
                wkr_value_inp = torch.cat([wtraj['feat'], wtraj['goal']], dim=-1)
                # ★ value は slow target net で計算（勾配不要）
                with torch.no_grad():
                    wkr_value = self._slow_value_wkr(wkr_value_inp).mode()
                
                # ★ lambda_return と advantage は no_grad の外で計算
                # dynamics/backprop モードでは reward→traj→action→actor の勾配パスが必要
                # REINFORCE モードでは adv.detach() するため影響なし
                wkr_target = tools.lambda_return(
                    wkr_reward,           # (k, N*B, 1) - r_0...r_{k-1}
                    wkr_value[:-1],       # (k, N*B, 1) - V(s_0)...V(s_{k-1}) (no_grad済み)
                    wtraj['cont'][1:],    # (k, N*B, 1) - next-state cont
                    bootstrap=wkr_value[-1],
                    lambda_=self._config.discount_lambda,
                    axis=0,
                )
                
                # ★ 公式実装に合わせた Weight 計算
                discount = self._config.discount
                wkr_weights = (torch.cumprod(wtraj['cont'], dim=0) / discount).detach()
                
                # ★ 公式 Director 準拠: return を std 正規化 → advantage を mean_std 正規化
                wkr_target_tensor = torch.stack(wkr_target, dim=1) if isinstance(wkr_target, (list, tuple)) else wkr_target
                normed_target_wkr = self.wkr_retnorm(wkr_target_tensor)
                normed_base_wkr = self.wkr_retnorm(wkr_value[:-1], update=False)
                adv_wkr = self.wkr_advnorm(normed_target_wkr - normed_base_wkr)
                
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
                    # 連続行動 (Normal分布): per-dimension entropy を [0,1] に正規化
                    # wkr_policy._dist.base_dist は Normal(mean, std) — std ∈ [min_std, max_std]
                    import math
                    base_dist = wkr_policy._dist.base_dist  # Normal per-dim
                    wkr_ent = base_dist.entropy()[:-1]  # (T, B, act_dim)
                    # Normal entropy = 0.5 * log(2*pi*e * std^2)
                    min_std = self.worker_actor._min_std
                    max_std = self.worker_actor._max_std
                    lo = 0.5 * math.log(2 * math.pi * math.e * min_std ** 2)
                    hi = 0.5 * math.log(2 * math.pi * math.e * max_std ** 2)
                    wkr_ent_normed = (wkr_ent - lo) / (hi - lo)  # per-dim [0, 1]
                    wkr_ent_loss, wkr_ent_mets = self.wkr_actent(wkr_ent_normed)
                    wkr_ent_loss = wkr_ent_loss.sum(-1)  # 次元ごとに合計
                else:
                    # 離散行動 (OneHot分布): scalar entropy を [0,1] に正規化
                    wkr_ent = wkr_policy.entropy()[:-1]
                    wkr_maxent = torch.log(torch.tensor(float(self._config.num_actions), device=wkr_ent.device))
                    wkr_ent_normed = wkr_ent / wkr_maxent
                    wkr_ent_loss, wkr_ent_mets = self.wkr_actent(wkr_ent_normed)
                # ★ 公式準拠: (REINFORCE_loss + entropy_loss) * weight
                wkr_actor_loss = (-wkr_actor_target + wkr_ent_loss.unsqueeze(-1)) * wkr_weights[:-1]
                wkr_actor_loss = torch.mean(wkr_actor_loss)
                
                # ========== Manager の訓練 ==========
                # Manager の報酬: extr + expl
                mgr_reward = (self.manager_rews["extr"] * mtraj['reward_extr'] + 
                              self.manager_rews["expl"] * mtraj['reward_expl'] + 
                              self.manager_rews["goal"] * mtraj['reward_goal'])
                
                # Manager の value 入力: feat のみ
                mgr_value_inp = mtraj['feat']
                # ★ value は slow target net で計算（勾配不要）
                with torch.no_grad():
                    mgr_value = self._slow_value_mgr(mgr_value_inp).mode()
                
                # ★ Manager も同様に lambda_return と advantage は no_grad 外
                mgr_target = tools.lambda_return(
                    mgr_reward,            # (n_abstract, B, 1)
                    mgr_value[:-1],        # (n_abstract, B, 1) - no_grad済み
                    mtraj['cont'][1:],     # (n_abstract, B, 1)
                    bootstrap=mgr_value[-1],
                    lambda_=self._config.discount_lambda,
                    axis=0,
                )
                
                # Weight 計算
                mgr_weights = (torch.cumprod(mtraj['cont'], dim=0) / discount).detach()
                
                # 正規化
                mgr_target_tensor = torch.stack(mgr_target, dim=1) if isinstance(mgr_target, (list, tuple)) else mgr_target
                normed_target_mgr = self.mgr_retnorm(mgr_target_tensor)
                normed_base_mgr = self.mgr_retnorm(mgr_value[:-1], update=False)
                adv_mgr = self.mgr_advnorm(normed_target_mgr - normed_base_mgr)
                    
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
        # print(f"Imagination and policy loss time: {e_time - s_time:.3f} sec")
        
        s_time = time.time()
        # ========== Value の更新 ==========
        with tools.RequiresGrad(self.worker_value), tools.RequiresGrad(self.manager_value):
            with torch.cuda.amp.autocast(self._use_amp):
                # Worker value loss
                # ★ 公式 VFunction.train 準拠: loss = -log_prob(target) * weight
                wkr_value_for_loss = self.worker_value(wkr_value_inp[:-1].detach())
                wkr_value_loss = -1 * wkr_value_for_loss.log_prob(wkr_target_tensor.detach())
                wkr_value_loss = torch.mean(wkr_weights[:-1] * wkr_value_loss[:, :, None])
                
                # Manager value loss
                # ★ 公式 VFunction.train 準拠: loss = -log_prob(target) * weight
                mgr_value_for_loss = self.manager_value(mgr_value_inp[:-1].detach())
                mgr_value_loss = -1 * mgr_value_for_loss.log_prob(mgr_target_tensor.detach())
                mgr_value_loss = torch.mean(mgr_weights[:-1] * mgr_value_loss[:, :, None])
        e_time = time.time()
        # print(f"Value loss time: {e_time - s_time:.3f} sec")
        
        s_time = time.time()
        # Metrics
        metrics.update(tools.tensorstats(wkr_value_for_loss.mode(), "worker_value"))
        metrics.update(tools.tensorstats(mgr_value_for_loss.mode(), "manager_value"))
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
        # Optimizer 更新（self 全体を RequiresGrad しない）
        with tools.RequiresGrad(self.manager_actor):
            metrics.update(self.mgr_actor_opt(mgr_actor_loss, self.manager_actor.parameters()))

        with tools.RequiresGrad(self.worker_actor):
            metrics.update(self.wkr_actor_opt(wkr_actor_loss, self.worker_actor.parameters()))

        with tools.RequiresGrad(self.manager_value):
            metrics.update(self.mgr_value_opt(mgr_value_loss, self.manager_value.parameters()))

        with tools.RequiresGrad(self.worker_value):
            metrics.update(self.wkr_value_opt(wkr_value_loss, self.worker_value.parameters()))
        
        # weights は wkr_weights を返す（既に正しく計算済み）
        e_time = time.time()
        return imag_feat, imag_state, imag_action, wkr_weights, metrics

    def _imagine(self, start, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        
        # Check if using VTA
        dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
        if dynamics_type == 'vta':
            return self._imagine_vta(start, horizon, dynamics)
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
        
        return feats, states, actions, goal_states, goal_latents
    
    def _imagine_vta(self, start, horizon, dynamics):
        """VTA imagination with jumpy or full modes."""
        imag_type = getattr(self._config, 'vta_imag_type', 'full')
        boundary_mode = getattr(self._config, 'vta_imag_boundary', 'prior')
        
        if imag_type == 'jumpy':
            # Jumpy imagination: each step is an abstract state transition
            def step(prev, _):
                state, _, _ = prev
                feat = dynamics.get_feat(state)
                inp = feat.detach()
                action = policy(inp).sample()
                succ = dynamics.jumpy_img_step(state, action)
                return succ, feat, action
        else:
            # Full imagination: normal timestep-level transitions
            def step(prev, _):
                state, _, _ = prev
                feat = dynamics.get_feat(state)
                inp = feat.detach()
                action = policy(inp).sample()
                succ = dynamics.img_step(state, action, boundary_mode=boundary_mode)
                return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, goal_state, extr_reward, expr_reward, goal_reward):
        # if "cont" in self._world_model.heads:
        #     inp = self._world_model.dynamics.get_feat(imag_state)
        #     discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        # else:
        # TODO: Directorの公式実装では，extr, expr, goalそれぞれについて，独立のMLPで価値推定して，最後に合算．
        # この実装では簡素化のため，価値関数は1つだけ用意し，3種類の報酬を重み付きで合算して学習．
        worker_reward = self.worker_rews["extr"] * extr_reward + self.worker_rews["expl"] * expr_reward + self.worker_rews["goal"] * goal_reward
        manager_reward = self.manager_rews['extr'] * extr_reward + self.manager_rews['expl'] * expr_reward + self.manager_rews['goal'] * goal_reward
        discount = self._config.discount * torch.ones_like(extr_reward)
        
        # 価値関数の推定
        # value = self.value(imag_feat).mode()
        # TODO: valueの時間長さはrewardと合わせるため，最初の時刻を削る
        wkr_value_inp = torch.cat([imag_feat, goal_state], dim=-1)  # (time, batch, feat_size + goal_size)
        mgr_value_inp = imag_feat
        worker_value = self.worker_value(wkr_value_inp).mode()[1:]
        manager_value = self.manager_value(mgr_value_inp).mode()[1:]
        
        # ★ split_traj は Actor/Value の計算全体に影響するため、
        # 現在は元の実装を使用。完全な実装には大幅なリファクタリングが必要。
        wkr_target = tools.lambda_return(
            worker_reward[1:],
            worker_value[:-1],
            discount[1:],
            bootstrap=worker_value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        
        # ★ split_traj を使用した実装（Actor loss 側も含めて調整が必要）
        # k = self._config.train_skill_duration
        # traj = {
        #     'feat': imag_feat,
        #     'goal': goal_state,
        #     'reward_worker': worker_reward,
        #     'cont': discount,
        # }
        # wtraj = self._split_traj(traj, k)
        # wkr_value_inp_split = torch.cat([wtraj['feat'], wtraj['goal']], dim=-1)
        # wkr_value_split = self.worker_value(wkr_value_inp_split).mode()
        # wkr_target = tools.lambda_return(
        #     wtraj['reward_worker'][1:],
        #     wkr_value_split[1:-1],
        #     wtraj['cont'][1:],
        #     bootstrap=wkr_value_split[-1],
        #     lambda_=self._config.discount_lambda,
        #     axis=0,
        # )
        
        mgr_target = tools.lambda_return(
            manager_reward[1:],
            manager_value[:-1],
            discount[1:],
            bootstrap=manager_value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return wkr_target, mgr_target, weights, worker_value[:-1], manager_value[:-1]
    
    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        goal_state,
        goal_latents,  # Goal AEのEncoder出力, 離散
        wkr_target,
        mgr_target,
        weights,
        wkr_value,
        mgr_value,
    ):
        metrics = {}
        wkr_inp = torch.cat([imag_feat, goal_state.detach()], dim=-1)
        mgr_inp = imag_feat.detach()
        wkr_policy = self.worker_actor(wkr_inp)
        mgr_policy = self.manager_actor(mgr_inp)

        # Q-val for actor is not transformed using symlog
        wkr_base = wkr_value
        mgr_base = mgr_value
        
        if self._config.reward_EMA:
            '''
            wkr_target = torch.stack(wkr_target, dim=1)  # (batch, time, 1)
            offset, scale = self.reward_ema(wkr_target, self.ema_vals)
            normed_target = (wkr_target - offset) / scale
            normed_base = (wkr_base - offset) / scale
            adv_wkr = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target_wkr"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

            mgr_target = torch.stack(mgr_target, dim=1)  # (batch, time, 1)
            offset, scale = self.reward_ema(mgr_target, self.ema_vals)
            normed_target = (mgr_target - offset) / scale  # 変更箇所 mgr用に新しく計算
            normed_base = (mgr_base - offset) / scale
            adv_mgr = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target_mgr"))
            '''
            # 変更箇所
            # Worker 用 EMA
            wkr_target = torch.stack(wkr_target, dim=1)  # (batch, time, 1)
            wkr_offset, wkr_scale = self.reward_ema(wkr_target, self.wkr_ema_vals)
            normed_target_wkr = (wkr_target - wkr_offset) / wkr_scale
            normed_base_wkr = (wkr_base - wkr_offset) / wkr_scale
            adv_wkr = normed_target_wkr - normed_base_wkr
            metrics.update(tools.tensorstats(normed_target_wkr, "normed_target_wkr"))
            metrics["EMA_005_wkr"] = to_np(self.wkr_ema_vals[0])
            metrics["EMA_095_wkr"] = to_np(self.wkr_ema_vals[1])

            # Manager 用 EMA
            mgr_target = torch.stack(mgr_target, dim=1)  # (batch, time, 1)
            mgr_offset, mgr_scale = self.reward_ema(mgr_target, self.mgr_ema_vals)
            normed_target_mgr = (mgr_target - mgr_offset) / mgr_scale
            normed_base_mgr = (mgr_base - mgr_offset) / mgr_scale
            adv_mgr = normed_target_mgr - normed_base_mgr
            metrics.update(tools.tensorstats(normed_target_mgr, "normed_target_mgr"))
            metrics["EMA_005_mgr"] = to_np(self.mgr_ema_vals[0])
            metrics["EMA_095_mgr"] = to_np(self.mgr_ema_vals[1])
        
        L, B = goal_latents.shape[:2]
        goal_latents = goal_latents.reshape(L, B, self._config.goal_enc_num_codebook, self._config.goal_enc_num_categorical)
        if self._config.imag_gradient == "dynamics":
            wkr_actor_target = adv_wkr
            mgr_actor_target = adv_mgr
        elif self._config.imag_gradient == "reinforce":
            wkr_actor_target = (
                wkr_policy.log_prob(imag_action)[1:-1][:, :, None]
                * (wkr_target - self.worker_value(wkr_inp[1:-1]).mode()).detach()
            )
            # goal_latents: (time, batch, num_codebook*num_categorical)
            mgr_actor_target = (
                mgr_policy.log_prob(goal_latents)[1:-1].sum(-1)[:, :, None]
                * (mgr_target - self.manager_value(mgr_inp[1:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            wkr_actor_target = (
                wkr_policy.log_prob(imag_action)[1:-1][:, :, None]
                * (wkr_target - self.worker_value(wkr_inp[1:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            wkr_actor_target = mix * wkr_target + (1 - mix) * wkr_actor_target
            metrics["imag_gradient_mix"] = mix
            
            mgr_actor_target = (
                mgr_policy.log_prob(goal_latents)[1:-1][:, :, None]
                * (mgr_target - self.manager_value(mgr_inp[1:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            mgr_actor_target = mix * mgr_target + (1 - mix) * mgr_actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        # actor_loss = -weights[:-1] * actor_target
        wkr_actor_loss = -weights[:-1] * wkr_actor_target
        mgr_actor_loss = -weights[:-1] * mgr_actor_target
        return wkr_actor_loss, mgr_actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                
                slow_wkr_value = self._slow_value_wkr
                wkr_value = self.worker_value
                for s, d in zip(wkr_value.parameters(), slow_wkr_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
                    
                slow_mgr_value = self._slow_value_mgr
                mgr_value = self.manager_value
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
                    # cont による重み付け: cumprod(cont[:-1]) で時間方向の割引
                    cont_seg = traj['cont'][start_idx:end_idx]
                    if cont_seg.shape[0] > 1:
                        weights = torch.cumprod(cont_seg[:-1], dim=0)  # (k-1, B, 1)
                        # 最初のステップの重みを1にする
                        weights = torch.cat([torch.ones_like(cont_seg[:1]), weights], dim=0)  # (k, B, 1)
                    else:
                        weights = torch.ones_like(seg)
                    weighted_sum = (seg * weights).sum(dim=0, keepdim=True)  # (1, B, 1)
                    weight_sum = weights.sum(dim=0, keepdim=True)  # (1, B, 1)
                    seg_mean = weighted_sum / weight_sum.clamp(min=1e-8)  # (1, B, 1)
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
            
    def take_action(self, feat, director_carry, training=False):
        """現在の状態から行動を選択する.
        Args:
            feat: 現在の特徴量表現, (time, dim_deter + dim_stoch)
            director_carry: Directorの内部状態 (辞書型)
        Returns:
            action: 選択された行動
            action_logprob: 選択された行動の対数確率
        """
        skill_duration = self._config.train_skill_duration if training else self._config.eval_skill_duration
        use_manager = director_carry["step"] % skill_duration == 0
        
        if use_manager:
            # manager policyを用いてGoalを推定.
            manager_inp = feat
            L = manager_inp.size(0)
            goal_dist = self.manager_actor(manager_inp)  # (batch, num_codebook, num_categorical)
            goal_latent = goal_dist.sample()  # (batch, num_codebook, num_categorical)
            goal_sample = goal_latent.reshape(L, -1)  # flatten to (batch, num_codebook * num_categorical)
            goal_state = self._world_model.decode_goal_ae_latent(goal_sample)  # (batch, dim_deter)
        else:
            goal_state = director_carry["goal"]
            goal_latent = director_carry["skill"]
        
        director_carry["step"] += 1
        director_carry["goal"] = goal_state
        director_carry["skill"] = goal_latent

        # worker policyを用いて行動を推定.
        worker_inp = torch.cat([feat, goal_state.detach()], dim=-1)
        worker_dist = self.worker_actor(worker_inp)
        action = worker_dist.sample()
        action_logprob = worker_dist.log_prob(action)
        return action, action_logprob, director_carry
        
