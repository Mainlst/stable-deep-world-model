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
                rec_loss = torch.mean((rec_states - org_states) ** 2)
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

        # Actorの更新
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
                
                # Exploration Rewardの計算, Eq. (6) in https://arxiv.org/pdf/2206.04114
                # Goal AEはReplay Bufferより訓練し，その再構成誤差によって状態の新規性を評価する．
                # 初期状態はManagerのGoal推定に依存しないため，計算から省く.
                # image_state['deter']: (time, batch, deter_dim)
                deter_states = imag_state['deter'][1:]
                L, B, D = deter_states.shape
                flat_states = deter_states.reshape(L * B, D)
                rec_states = self._world_model.reconstruct_goal_ae(flat_states)  # (L*B, D)
                rec_states = rec_states.reshape(L, B, D)
                expl_reward = torch.sum((rec_states - deter_states) ** 2, dim=-1, keepdim=True) 
                # expl_reward: (time-1, batch, 1)
                
                # Goal Rewardの計算
                flatten_goal_states = goal_states[1:].reshape(L*B, -1)
                goal_reward = self._cosine_max_similarity(
                    flat_states,
                    flatten_goal_states,
                ).reshape(L, B, 1)
                # goal_reward: (time-1, batch, 1)
                
                # 方策のエントロピー計算
                manager_inp = imag_feat
                worker_inp = torch.cat([imag_feat, goal_states.detach()], dim=-1)
                manager_ent = self.manager_actor(manager_inp).entropy()  # (time, batch, num_codebook)
                worker_ent = self.worker_actor(worker_inp).entropy()  # (time, batch)
                
                # Get state entropy (VTA returns different structure)
                dynamics_type = getattr(self._config, 'dynamics_type', 'rssm')
                if dynamics_type == 'vta':
                    # For VTA, use observation level distribution
                    state_ent = self._world_model.dynamics.get_dist(imag_state, level="obs").entropy()
                else:
                    state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                    
                # this target is not scaled by ema or sym_log.
                wkr_target, mgr_target, weights, wkr_value, mgr_value = self._compute_target(
                    imag_feat, imag_state, goal_states, extr_reward, expl_reward, goal_reward
                )
                
                # TODO: 損失の計算
                wkr_actor_loss, mgr_actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    goal_states,
                    goal_latents,
                    wkr_target,
                    mgr_target,
                    weights,
                    wkr_value,
                    mgr_value,
                )
                wkr_actor_loss = wkr_actor_loss - self._config.actor["entropy"] * worker_ent[1:-1, ..., None]
                wkr_actor_loss = torch.mean(wkr_actor_loss)

                mgr_actor_loss = mgr_actor_loss - self._config.actor["entropy"] * manager_ent[1:-1].sum(dim=-1, keepdim=True)
                mgr_actor_loss = torch.mean(mgr_actor_loss)
                
                wkr_value_inp = torch.cat([imag_feat, goal_states], dim=-1)
                mgr_value_inp = imag_feat
        
        # Valueの更新
        with tools.RequiresGrad(self.worker_value), tools.RequiresGrad(self.manager_value):
            with torch.cuda.amp.autocast(self._use_amp):
                # value = self.value(value_input[:-1].detach())
                wkr_value = self.worker_value(wkr_value_inp[1:-1].detach())
                mgr_value = self.manager_value(mgr_value_inp[1:-1].detach())
                
                wkr_target = torch.stack(wkr_target, dim=1)
                mgr_target = torch.stack(mgr_target, dim=1)
                
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                wkr_value_loss = -1 * wkr_value.log_prob(wkr_target.detach())
                mgr_value_loss = -1 * mgr_value.log_prob(mgr_target.detach())
                wkr_slow_target = self._slow_value_wkr(wkr_value_inp[1:-1].detach())
                mgr_slow_target = self._slow_value_mgr(mgr_value_inp[1:-1].detach())
                if self._config.critic["slow_target"]:
                    wkr_value_loss = wkr_value_loss - wkr_value.log_prob(wkr_slow_target.mode().detach())
                    mgr_value_loss = mgr_value_loss - mgr_value.log_prob(mgr_slow_target.mode().detach())

                # (time, batch, 1), (time, batch, 1) -> (1,)
                wkr_value_loss = torch.mean(weights[:-1] * wkr_value_loss[:, :, None])
                mgr_value_loss = torch.mean(weights[:-1] * mgr_value_loss[:, :, None])

        metrics.update(tools.tensorstats(wkr_value.mode(), "worker_value"))
        metrics.update(tools.tensorstats(mgr_value.mode(), "manager_value"))
        metrics.update(tools.tensorstats(extr_reward, "imag_reward"))
        metrics.update(tools.tensorstats(expl_reward, "imag_expl_reward"))
        metrics.update(tools.tensorstats(goal_reward, "imag_goal_reward"))
        metrics.update(mets)
        
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

        with tools.RequiresGrad(self):
            metrics.update(self.mgr_actor_opt(mgr_actor_loss, self.manager_actor.parameters()))
            metrics.update(self.wkr_actor_opt(wkr_actor_loss, self.worker_actor.parameters()))
            metrics.update(self.mgr_value_opt(mgr_value_loss, self.manager_value.parameters()))
            metrics.update(self.wkr_value_opt(wkr_value_loss, self.worker_value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

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
        """RSSM imagination with Manager & Actor"""
        def step(prev, state_idx):
            
            # 今の状態の取得
            state, _, _, prev_goal_state, prev_goal_latent = prev
            feat = dynamics.get_feat(state)  # concat [stoch, deter]
            inp = feat.detach()
            
            # Manager PolicyによるGoalの推定
            # use_manager: Managerを用いるかどうか，Kステップに一度Goalを更新する
            if state_idx % self._config.train_skill_duration == 0:
                B = inp.size(0)
                goal_dist = self.manager_actor(inp)  # (batch, num_codebook, num_categorical)
                goal_sample = goal_dist.sample()  # (batch, num_codebook, num_categorical)
                # Goal DecoderによるGoal状態の再構成
                goal_sample = goal_sample.reshape(B, -1)  # flatten to (batch, num_codebook * num_categorical)
                goal_state = self._world_model.decode_goal_ae_latent(goal_sample)  # (batch, dim_deter)
                goal_latents = goal_sample
            else:
                goal_state = prev_goal_state  # 前回のGoal状態を使用
                goal_latents = prev_goal_latent
            
            # Worker PolicyによるActionの推定
            worker_inp = torch.cat([inp, goal_state.detach()], dim=-1)  # (batch, feat_size + goal_size)
            action_dist = self.worker_actor(worker_inp)
            action = action_dist.sample()  # (batch, action_dim)

            # 世界モデルによる状態遷移
            succ = dynamics.img_step(state, action)
            return succ, feat, action, goal_state, goal_latents

        succ, feats, actions, goal_states, goal_latents = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
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
        
        wkr_target = tools.lambda_return(
            worker_reward[1:],
            worker_value[:-1],
            discount[1:],
            bootstrap=worker_value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
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
        
