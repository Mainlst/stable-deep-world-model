import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PINPAD_SRC = Path(__file__).resolve().parents[2] / "tasks" / "pinpad" / "src"
if str(PINPAD_SRC) not in sys.path:
    sys.path.append(str(PINPAD_SRC))

from pinpad import PinPad


class PinPadEnv(gym.Env):
    def __init__(self, layout: str):
        super().__init__()
        self._env = PinPad.make(layout=layout)
        obs, _ = self._env.reset()
        self._obs_shape = obs.shape
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=self._obs_shape, dtype=np.uint8
                ),
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            }
        )
        self.action_space = self._env.action_space
        self.action_space.discrete = True

    def reset(self, **kwargs):
        obs, _ = self._env.reset()
        return {
            "image": obs,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        obs = {
            "image": obs,
            "is_first": False,
            "is_last": done,
            "is_terminal": bool(terminated),
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
