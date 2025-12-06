"""
デフォルトとなる環境設定の定義です．以下のように使用します．
from stable_deep_world_model.src_vta.envs.config import DefaultEnvConfig

(i) そのまま利用する場合
config = DefaultEnvConfig()
(ii) 一部のパラメータを変更する場合
config = DefaultEnvConfig(num_envs=16, img_size=(128, 128))
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class DefaultEnvConfig:
    env_id: str = "CartPole-v1"  # または "dm_control/cheetah-run"
    num_envs: int = 8            # 並列環境数 (Batch size)
    img_size: Tuple[int, int] = (64, 64)
    frame_stack: int = 4     # 観測のフレームスタック数
    action_repeat: int = 2       # Action Repeat
    seed: int = 42
    save_video: bool = True
    video_interval: int = 50     # 何エピソードごとにGIFを保存するか
    render_mode: str = "rgb_array"
    gray_scale: bool = False