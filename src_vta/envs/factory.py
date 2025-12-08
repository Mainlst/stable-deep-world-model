import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import numpy as np

from .config import DefaultEnvConfig
from .wrappers.recoder import EpisodeRecoderWrapper
from .wrappers.preprocessing import ActionRepeatWrapper, ChannelFirstWrapper

def make_single_env(cfg: DefaultEnvConfig, rank: int, output_dir: str):
    """
    単一の環境Wrapperを作成します．
    """
    def _trunk():
        # 1. 環境の作成
        env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
        
        # 2. 行動の反復（Action Repeat）
        if cfg.action_repeat > 1:
            env = ActionRepeatWrapper(env, repeat=cfg.action_repeat)
        
        # 3. 観測の前処理（リサイズ，グレースケール化，フレームスタック）
        env = ResizeObservation(env, shape=cfg.img_size)
        if cfg.gray_scale:
            env = GrayscaleObservation(env, keep_dim=True)
        
        
        # 4. Recoder Wrapper
        if cfg.save_video:
            env = EpisodeRecoderWrapper(
                env,
                save_dir=output_dir,
                rank=rank,
                save_interval=cfg.video_interval
            )
        
        # 5. Channel-first 変換
        env = ChannelFirstWrapper(env)
        
        # 6. Frame Stack
        if cfg.frame_stack > 1:
            env = FrameStackObservation(env, stack_size=cfg.frame_stack)
        
        env.reset(seed=cfg.seed + rank)
        return env
    return _trunk

def build_vector_env(cfg: DefaultEnvConfig, output_dir: str, shared_memory: bool = False, use_sync: bool = True):
    """
    並列環境を構築します．
    Usage: 
    from envs.config import DefaultEnvConfig
    from envs.factory import build_vector_env
    
    cfg = DefaultEnvConfig()
    envs = build_vector_env(cfg)
    obs, info = envs.reset()
    
    try:
        for step in range(1000):
            actions = envs.action_space.sample()
            next_obs, rewards, term, trunc, info = envs.step(actions)
            ...
            obs = next_obs
    except KeyboardInterrupt: 
        pass
    finally:
        envs.close()    
    
    """
    env_fns = [make_single_env(cfg, rank, output_dir) for rank in range(cfg.num_envs)]
    if use_sync:
        vector_env = SyncVectorEnv(env_fns)
    else:
        vector_env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    return vector_env


        
    