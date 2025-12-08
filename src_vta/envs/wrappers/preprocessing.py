import gymnasium as gym
from gymnasium.spaces import Box

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        """Action Repeat Wrapper
        環境のステップを複数回繰り返し,累積報酬と終了時の観測/終了/中断状態を返します.
        """
        super().__init__(env)
        self.repeat = repeat
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.repeat):
            obs, reward, term, trunc = self.env.step(action)
            total_reward += reward
            terminated = term
            truncated = trunc
            
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
    
class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """
        観測(B, H, W, C) を (B, C, H, W) に変換するラッパー
        観測のみを書き換えるため，ObservationWrapperを継承しています．
        """
        super().__init__(env)
        
        old_shape = env.observation_space.shape
        dtype = env.observation_space.dtype
        
        assert len(old_shape) == 3, "Observation must be an image with shape (H, W, C)"
        new_shape = (old_shape[2], old_shape[0], old_shape[1])  # (C, H, W)
        self.observation_space = Box(
            low=0,
            high=255 if dtype == 'uint8' else 1.0,
            shape=new_shape,
            dtype=dtype
        )
    
    def observation(self, observation):
        return observation.transpose(2, 0, 1)  # HWC -> CHW
        