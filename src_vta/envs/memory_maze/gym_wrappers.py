from typing import Any, Tuple
import numpy as np

import dm_env
import gymnasium as gym
from dm_env import specs
from gymnasium import spaces


class GymWrapper(gym.Env):

    def __init__(self, env: dm_env.Environment):
        super().__init__()
        self.env = env
        self.action_space = _convert_to_space(env.action_spec())
        self.observation_space = _convert_to_space(env.observation_spec())

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        # Gymnasium style reset: returns (obs, info)
        # dm_env.Environment にシードを渡したい場合はここで処理
        if seed is not None:
            # env にシードメソッドがあるなら呼ぶ
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        ts = self.env.reset()
        obs = ts.observation
        info: dict = {}
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        # Gymnasium style step: returns (obs, reward, terminated, truncated, info)
        ts = self.env.step(action)
        assert not ts.first(), "dm_env.step() caused reset, reward will be undefined."
        assert ts.reward is not None

        obs = ts.observation
        reward = float(ts.reward)

        terminated = ts.last() and ts.discount == 0.0      # 真の終端
        truncated = ts.last() and ts.discount != 0.0       # TimeLimit などによる打ち切りを truncated 扱い

        info: dict = {}
        if truncated:
            # Gym 時代の TimeLimit.truncated 相当。必要なら残しておく。
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info


def _convert_to_space(spec: Any) -> gym.Space:
    # Inverse of acme.gym_wrappers._convert_to_spec

    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)

    if isinstance(spec, specs.BoundedArray):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=spec.minimum.item() if len(spec.minimum.shape) == 0 else spec.minimum,
            high=spec.maximum.item() if len(spec.maximum.shape) == 0 else spec.maximum,
        )

    if isinstance(spec, specs.Array):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=-np.inf,
            high=np.inf,
        )

    if isinstance(spec, tuple):
        return spaces.Tuple(tuple(_convert_to_space(s) for s in spec))

    if isinstance(spec, dict):
        return spaces.Dict({key: _convert_to_space(value) for key, value in spec.items()})

    raise ValueError(f"Unexpected spec: {spec}")