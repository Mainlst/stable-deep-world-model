import torch
from torch import nn
from torch import distributions as torchd
import numpy as np
import functools

from . import networks
from . import tools
from . import models

to_np = lambda x: x.detach().cpu().numpy()

class AutoAdapt(nn.Module):
    def __init__(
        self,
        shape=(),
        scale=1.0,
        target=10.0,
        min=1e-5,
        max=1.0,
        vel=0.1,
    ):
        super(AutoAdapt, self).__init__()
        self.shape = shape
        self.scale = scale
        self.target = target
        self.min = min
        self.max = max
        self.vel = vel
        self.register_buffer("param", torch.zeros(shape) + scale)

    def forward(self, value):
        scale = torch.clamp(self.param, self.min, self.max)
        loss = scale.detach() * value
        
        # Exponential update similar to official
        new_scale = scale * torch.exp(self.vel * (value.detach().mean() - self.target))
        new_scale = torch.clamp(new_scale, self.min, self.max)
        self.param.copy_(new_scale)
        
        return loss, {"kl_scale": scale.detach().mean(), "kl_raw": value.detach().mean()}

class GoalAutoEncoder(nn.Module):
    def __init__(self, config):
        super(GoalAutoEncoder, self).__init__()
        self._config = config
        
        # Skill shape
        if hasattr(config, 'skill_shape'):
             self._skill_shape = tuple(config.skill_shape)
        else:
             self._skill_shape = (config.num_actions,) 

        # Goal is the stochastic state
        self._goal_size = config.dyn_stoch
        if config.dyn_discrete:
             self._goal_size *= config.dyn_discrete
        
        # Context is 'deter'
        self._context_size = config.dyn_deter

        # Encoder: (goal, context) -> skill
        enc_inp = self._goal_size + self._context_size
        self.enc = networks.MLP(
            enc_inp,
            self._skill_shape,
            layers=config.goal_ae["layers"],
            units=config.goal_ae["units"],
            act=config.act,
            norm=config.norm,
            dist=config.goal_ae["enc_dist"],
            outscale=config.goal_ae.get("outscale", 1.0),
            name="GoalEnc"
        )

        # Decoder: (skill, context) -> goal
        skill_size = int(np.prod(self._skill_shape))
        dec_inp = skill_size + self._context_size
        self.dec = networks.MLP(
             dec_inp,
             (self._goal_size,), 
             layers=config.goal_ae["layers"],
             units=config.goal_ae["units"],
             act=config.act,
             norm=config.norm,
             dist=config.goal_ae["dec_dist"],
             outscale=config.goal_ae.get("outscale", 1.0),
             name="GoalDec"
        )
        
        self.kl = AutoAdapt(**config.encdec_kl) if hasattr(config, 'encdec_kl') else AutoAdapt(target=10.0)
        
        self.opt = tools.Optimizer(
            "goal_ae",
            self.parameters(),
            config.goal_ae["lr"],
            config.goal_ae["eps"],
            config.goal_ae["grad_clip"],
            wd=config.weight_decay,
            opt=config.opt,
            use_amp=config.precision == 16,
        )

    def forward(self, goal, context, skill=None):
        enc_in = torch.cat([goal, context], dim=-1)
        post = self.enc(enc_in)
        if skill is None:
            skill = post.sample()
        
        skill_flat = skill.reshape(skill.shape[:-len(self._skill_shape)] + (-1,))
        dec_in = torch.cat([skill_flat, context], dim=-1)
        recon_dist = self.dec(dec_in)
        
        return post, recon_dist, skill

    def train_vae_replay(self, data, world_model):
        metrics = {}
        with torch.no_grad():
             data = world_model.preprocess(data)
             embed = world_model.encoder(data)
             post, _ = world_model.dynamics.observe(embed, data['action'], data['is_first'])
        
        k = self._config.director_duration
        T = post['deter'].shape[1]
        
        if T <= k: return {}
        
        # Context: deter
        context = post['deter'][:, :-k]
        
        # Goal: stoch
        stoch = post['stoch']
        if self._config.dyn_discrete:
            stoch = stoch.reshape(stoch.shape[:-2] + (-1,))
        goal_stoch = stoch[:, k:]
        
        # Flatten
        context = context.reshape(-1, context.shape[-1])
        goal_stoch = goal_stoch.reshape(-1, goal_stoch.shape[-1])
        
        with tools.RequiresGrad(self):
             with torch.cuda.amp.autocast(self._config.precision == 16):
                  post_dist, recon_dist, skill = self(goal_stoch, context)
                  lik = recon_dist.log_prob(goal_stoch)
                  
                  if self._config.goal_kl:
                       ent = post_dist.entropy()
                       kl = -ent
                       kl_loss, mets = self.kl(kl)
                       metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
                  else:
                       kl_loss = 0.0
                  
                  loss = (-lik + kl_loss).mean()
        
        metrics.update(self.opt(loss, self.parameters()))
        metrics['goal_ae_loss'] = to_np(loss)
        metrics['goal_ae_recon'] = to_np(lik.mean())
        return metrics

class DirectorBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(DirectorBehavior, self).__init__()
        self._config = config
        self._world_model = world_model
        
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self._feat_size = feat_size
        self._skill_shape = tuple(getattr(config, 'skill_shape', (8, 8)))
        
        self._goal_size = config.dyn_stoch
        if config.dyn_discrete:
             self._goal_size *= config.dyn_discrete

        self.manager_actor = networks.MLP(
            feat_size,
            self._skill_shape,
            config.manager["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.manager["dist"],
            name="ManagerActor"
        )
        self.manager_value = networks.MLP(
             feat_size,
             (255,) if config.critic["dist"] == "symlog_disc" else (),
             config.critic["layers"],
             config.units,
             config.act,
             config.norm,
             dist=config.critic["dist"],
             name="ManagerValue"
        )

        worker_inp = feat_size + self._goal_size
        self.worker_actor = networks.MLP(
            worker_inp,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.actor["dist"],
            name="WorkerActor"
        )
        self.worker_value = networks.MLP(
             worker_inp,
             (255,) if config.critic["dist"] == "symlog_disc" else (),
             config.critic["layers"],
             config.units,
             config.act,
             config.norm,
             dist=config.critic["dist"],
             name="WorkerValue"
        )

        self.goal_ae = GoalAutoEncoder(config)
        
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=config.precision == 16)
        self._manager_actor_opt = tools.Optimizer(
            "manager_actor", self.manager_actor.parameters(), config.manager["lr"], config.actor["eps"], config.actor["grad_clip"], **kw
        )
        self._manager_value_opt = tools.Optimizer(
            "manager_value", self.manager_value.parameters(), config.manager["lr"], config.critic["eps"], config.critic["grad_clip"], **kw
        )
        self._worker_actor_opt = tools.Optimizer(
            "worker_actor", self.worker_actor.parameters(), config.actor["lr"], config.actor["eps"], config.actor["grad_clip"], **kw
        )
        self._worker_value_opt = tools.Optimizer(
            "worker_value", self.worker_value.parameters(), config.critic["lr"], config.critic["eps"], config.critic["grad_clip"], **kw
        )
        
        self.reward_ema = models.RewardEMA(device=config.device)

    def policy(self, feat, state=None, training=True):
        stoch_dim = self._goal_size
        state_stoch = feat[..., :stoch_dim]
        state_deter = feat[..., stoch_dim:]
        
        if state is None:
             batch_size = feat.shape[0]
             skill_dist = self.manager_actor(feat)
             skill = skill_dist.sample() if training else skill_dist.mode()
             skill_flat = skill.reshape(skill.shape[0], -1)
             dec_in = torch.cat([skill_flat, state_deter], -1)
             goal = self.goal_ae.dec(dec_in).mode()
             if getattr(self._config, 'manager_delta', False):
                  goal = state_stoch + goal
             
             step = torch.zeros(batch_size, device=feat.device, dtype=torch.long)
             state = {'skill': skill, 'goal': goal, 'step': step}
        
        skill = state['skill']
        goal = state['goal']
        step = state['step']
        
        duration = self._config.director_duration
        switch = (step % duration == 0)
        
        if torch.any(switch):
             skill_dist = self.manager_actor(feat)
             new_skill = skill_dist.sample() if training else skill_dist.mode()
             new_skill_flat = new_skill.reshape(new_skill.shape[0], -1)
             dec_in = torch.cat([new_skill_flat, state_deter], -1)
             new_goal = self.goal_ae.dec(dec_in).mode()
             if getattr(self._config, 'manager_delta', False):
                  new_goal = state_stoch + new_goal
             
             
             mask = switch.float()
             mask = mask.view(mask.shape[0], *([1] * (skill.ndim - 1)))
             skill = mask * new_skill + (1-mask) * skill
             goal = mask * new_goal + (1-mask) * goal

        worker_in = torch.cat([feat, goal], dim=-1)
        actor_dist = self.worker_actor(worker_in)
        step = step + 1
        new_state = {'skill': skill, 'goal': goal, 'step': step}
        
        return actor_dist, new_state

    def _train(self, start, objective, data=None):
        metrics = {}
        if data is not None and self._config.vae_replay:
             metrics.update(self.goal_ae.train_vae_replay(data, self._world_model))

        with tools.RequiresGrad(self):
             with torch.cuda.amp.autocast(self._config.precision == 16):
                  traj_feat, traj_state, traj_action, traj_director_state = self._imagine_hierarchy(
                      start, self._config.imag_horizon
                  )
                  
                  reward = objective(traj_feat, traj_state, traj_action)
                  goal_reward = self._goal_reward(traj_state, traj_director_state)
                  metrics['goal_reward_mean'] = to_np(goal_reward.mean())
                  
                  if self._config.expl_rew == 'adver':
                       expl_reward = self._elbo_reward(traj_feat, traj_state['deter'], traj_director_state)
                       metrics['expl_reward_mean'] = to_np(expl_reward.mean())
                  else:
                       expl_reward = torch.zeros_like(reward)

                  worker_reward = torch.zeros_like(reward)
                  if self._config.worker_rews.get('extr', 0.0) > 0:
                      worker_reward += self._config.worker_rews['extr'] * reward
                  if self._config.worker_rews.get('goal', 0.0) > 0:
                      worker_reward += self._config.worker_rews['goal'] * goal_reward
                  if self._config.worker_rews.get('expl', 0.0) > 0:
                       worker_reward += self._config.worker_rews['expl'] * expl_reward
                  
                  manager_reward = torch.zeros_like(reward)
                  if self._config.manager_rews.get('extr', 0.0) > 0:
                      manager_reward += self._config.manager_rews['extr'] * reward
                  if self._config.manager_rews.get('goal', 0.0) > 0:
                      manager_reward += self._config.manager_rews['goal'] * goal_reward
                  if self._config.manager_rews.get('expl', 0.0) > 0:
                      manager_reward += self._config.manager_rews['expl'] * expl_reward
                  
                  wtraj_feat, wtraj_action, wtraj_reward, wtraj_disc = self._split_traj(
                      traj_feat, traj_action, worker_reward, traj_director_state
                  )
                  
                  metrics.update(self._update_worker(wtraj_feat, wtraj_action, wtraj_reward, wtraj_disc, traj_director_state))
                  
                  mtraj_feat, mtraj_action, mtraj_reward, mtraj_disc = self._abstract_traj(
                      traj_feat, traj_director_state['skill'], manager_reward
                  )
                  
                  metrics.update(self._update_manager(mtraj_feat, mtraj_action, mtraj_reward, mtraj_disc))

        return traj_feat, traj_state, traj_action, None, metrics

    def _imagine_hierarchy(self, start, horizon):
         dynamics = self._world_model.dynamics
         flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
         start = {k: flatten(v) for k, v in start.items()}
         
         def step(prev, _):
             state, _, _, director_state = prev
             feat = dynamics.get_feat(state)
             actor_dist, new_director_state = self.policy(feat, director_state)
             action = actor_dist.sample()
             succ = dynamics.img_step(state, action)
             return succ, feat, action, new_director_state

         feat = dynamics.get_feat(start)
         stoch_dim = self._goal_size
         state_stoch = feat[..., :stoch_dim]
         state_deter = feat[..., stoch_dim:]
         
         batch_size = feat.shape[0]
         skill = self.manager_actor(feat).sample()
         skill_flat = skill.reshape(skill.shape[0], -1)
         dec_in = torch.cat([skill_flat, state_deter], -1)
         goal = self.goal_ae.dec(dec_in).mode()
         
         if getattr(self._config, 'manager_delta', False):
              goal = state_stoch + goal
         
         step_cnt = torch.zeros(batch_size, device=feat.device, dtype=torch.long)
         init_director_state = {'skill': skill, 'goal': goal, 'step': step_cnt}

         succ, feats, actions, director_states = tools.static_scan(
             step, [torch.arange(horizon)], (start, None, None, init_director_state)
         )
         
         states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
         return feats, states, actions, director_states

    def _split_traj(self, feat, action, reward, director_state):
        k = self._config.director_duration
        T, B = feat.shape[:2]
        window_size = k + 1
        stride = k
        if T < window_size:
             return feat, action, reward, torch.ones_like(reward)

        def make_chunks(x):
             x_flat = x.reshape(T, -1)
             chunks = x_flat.unfold(0, window_size, stride)
             chunks = chunks.permute(2, 0, 1)
             out = chunks.reshape(window_size, -1, *shape[2:])
             return out

        shape = feat.shape
        w_feat = make_chunks(feat)
        w_action = make_chunks(action)
        w_reward = make_chunks(reward)
        goals = director_state['goal']
        w_goal = make_chunks(goals)
        
        w_feat_full = torch.cat([w_feat, w_goal], -1)
        w_disc = self._config.discount * torch.ones_like(w_reward)
        
        return w_feat_full, w_action, w_reward, w_disc

    def _abstract_traj(self, feat, skill, reward):
        k = self._config.director_duration
        T, B = feat.shape[:2]
        window_size = k
        stride = k
        if T < window_size:
             return feat, skill, reward, torch.ones_like(reward)

        def make_chunks(x):
             x_flat = x.reshape(T, -1)
             chunks = x_flat.unfold(0, window_size, stride)
             chunks = chunks.permute(2, 0, 1)
             out = chunks.reshape(window_size, -1, *shape[2:])
             return out

        shape = feat.shape
        chunks_feat = make_chunks(feat)
        chunks_skill = make_chunks(skill)
        chunks_reward = make_chunks(reward)
        m_feat = chunks_feat[0] 
        m_action = chunks_skill[0]
        m_reward = chunks_reward.mean(0)
        m_disc = (self._config.discount ** k) * torch.ones_like(m_reward)
        
        return m_feat, m_action, m_reward, m_disc

    def _update_worker(self, feat, action, reward, disc, director_state):
        metrics = {}
        target, weights, base = self._compute_target(
             self.worker_value, feat, reward, disc
        )
        actor_loss, mets = self._compute_actor_loss(
             self.worker_actor, feat, action, target, weights, base
        )
        metrics.update({f'worker_{k}': v for k, v in mets.items()})
        value_loss, mets = self._compute_value_loss(
             self.worker_value, feat, target, weights, self._worker_value_opt
        )
        metrics.update({f'worker_{k}': v for k, v in mets.items()})
        metrics.update(self._worker_actor_opt(actor_loss, self.worker_actor.parameters()))
        return metrics

    def _update_manager(self, feat, action, reward, disc):
        metrics = {}
        target, weights, base = self._compute_target(
             self.manager_value, feat, reward, disc
        )
        actor_loss, mets = self._compute_actor_loss(
             self.manager_actor, feat, action, target, weights, base
        )
        metrics.update({f'manager_{k}': v for k, v in mets.items()})
        value_loss, mets = self._compute_value_loss(
             self.manager_value, feat, target, weights, self._manager_value_opt
        )
        metrics.update({f'manager_{k}': v for k, v in mets.items()})
        metrics.update(self._manager_actor_opt(actor_loss, self.manager_actor.parameters()))
        return metrics

    def _compute_target(self, value_net, feat, reward, disc):
        value = value_net(feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            disc[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(self, actor_net, feat, action, target, weights, base):
        metrics = {}
        inp = feat[:-1].detach()
        policy = actor_net(inp)
        target = torch.stack(target, dim=1)
        base = base.detach()
        if self._config.reward_EMA:
             offset, scale = self.reward_ema(target, self.reward_ema.ema_vals)
             normed_target = (target - offset) / scale
             normed_base = (base - offset) / scale
             adv = normed_target - normed_base
        else:
             adv = target - base
        action_inp = action[:-1]
        lp = policy.log_prob(action_inp)
        actor_loss = -weights[:-1] * lp.unsqueeze(-1) * adv.detach()
        ent = policy.entropy()
        actor_loss -= self._config.actor["entropy"] * ent.unsqueeze(-1)
        return actor_loss.mean(), metrics

    def _compute_value_loss(self, value_net, feat, target, weights, optimizer):
         metrics = {}
         inp = feat[:-1].detach()
         value = value_net(inp)
         target = torch.stack(target, dim=1)
         loss = -value.log_prob(target.detach())
         loss = (weights[:-1] * loss.unsqueeze(-1)).mean()
         return loss, metrics

    def _goal_reward(self, state, director_state):
        stoch = state['stoch']
        if self._config.dyn_discrete:
             stoch = stoch.reshape(stoch.shape[:-2] + (-1,))
        goal = director_state['goal']
        feat = stoch
        mode = self._config.goal_reward
        if mode == 'squared':
             return -torch.sum((feat - goal) ** 2, dim=-1)
        elif mode == 'norm':
             return -torch.linalg.norm(feat - goal, dim=-1)
        elif mode == 'cosine':
             return torch.nn.functional.cosine_similarity(feat, goal, dim=-1)
        elif mode == 'cosine_max':
             gnorm = torch.linalg.norm(goal, dim=-1, keepdim=True) + 1e-12
             fnorm = torch.linalg.norm(feat, dim=-1, keepdim=True) + 1e-12
             norm = torch.maximum(gnorm, fnorm)
             return torch.sum((goal / norm) * (feat / norm), dim=-1)
        raise NotImplementedError(f"Goal reward {mode} not implemented")

    def _elbo_reward(self, feat, deter, director_state):
        skill = director_state['skill']
        context = deter 
        stoch_dim = self._goal_size
        stoch = feat[..., :stoch_dim] 
        goal_feat = stoch
        post_dist, recon_dist, _ = self.goal_ae(goal_feat, context)
        ll = recon_dist.log_prob(goal_feat)
        ent = post_dist.entropy()
        kl = -ent
        kl_scale = 1.0 
        if hasattr(self.goal_ae.kl, 'param'):
             kl_scale = self.goal_ae.kl.param.detach()
        return (kl - ll / kl_scale).mean(-1)
