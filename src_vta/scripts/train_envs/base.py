from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src_vta.models import VTA


@dataclass
class EnvTrainContext:
    """共通トレーナーが使うランタイム情報。"""

    train_provider: "TrainBatchProvider"
    test_loader: DataLoader
    eval_obs: torch.Tensor
    eval_act: torch.Tensor
    action_size: int
    use_amp: bool
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: Optional[int] = None


class TrainBatchProvider:
    """バッチ供給元を環境ごとに隠蔽するためのインターフェース。"""

    def next_batch(self, step: int):
        raise NotImplementedError


class EnvironmentAdapter:
    """環境固有の処理をトレーナーに差し込むためのアダプター。"""

    name: str

    def configure_args(self, args):
        return args

    def build_context(self, args) -> EnvTrainContext:
        raise NotImplementedError

    def process_batch(self, batch, args) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def log_metrics(self, mode: str, results, writer, step: int, wandb_run=None) -> float:
        raise NotImplementedError

    def on_best_model(self, model: VTA, ctx: EnvTrainContext, args, step: int):
        raise NotImplementedError
