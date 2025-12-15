import glob
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src_vta.data.paths import resolve_maze_data_dir
from src_vta.utils import preprocess, visualize_results
from .base import EnvironmentAdapter, EnvTrainContext, TrainBatchProvider

ACTION_SIZE = 3


class MazeDataset(torch.utils.data.Dataset):
    """3D Maze npz を遅延ロードするデータセット。"""

    def __init__(
        self,
        length: int,
        partition: str = "train",
        image_width: int = 32,
        image_height: int = 32,
        image_channels: int = 3,
        one_hot_action: bool = True,
        data_dir: Optional[Union[Path, str]] = None,
    ):
        self.data_dir = resolve_maze_data_dir(data_dir)
        self.partition = partition
        self.length = length
        self.height = image_height
        self.width = image_width
        self.image_channels = image_channels
        self.one_hot_action = one_hot_action
        self.action_size = ACTION_SIZE

        dir_path = self.data_dir / self.partition
        self.file_paths = sorted(glob.glob(str(dir_path / "*.npz")))

        if len(self.file_paths) == 0:
            print(
                f"エラー: {dir_path} に .npz ファイルが見つかりません\n"
                "  - データを data/3d_maze_default/{train,test} に配置してください。\n"
                "  - もしくは Config.maze_data_dir で場所を指定し、"
                "`python -m src_vta.data.generate_npz --out <path>` で生成できます。"
            )
            sys.exit(1)

        print(
            f"データセット ({self.partition}): {len(self.file_paths)} エピソードを検出（遅延ロード）"
        )

    def _resize(self, frames: np.ndarray):
        return np.stack([cv2.resize(frame, (self.width, self.height)) for frame in frames])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        with np.load(path, allow_pickle=True) as sample_episode:
            frames_raw = sample_episode["video"]
            actions_raw = sample_episode["actions"]

        frames = self._resize(frames_raw)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        actions = torch.from_numpy(actions_raw)

        seq_len = frames.shape[0]
        start_t = np.random.randint(0, max(seq_len - self.length, 1))

        frames = frames[start_t : start_t + self.length]
        actions = actions[start_t : start_t + self.length]

        if self.one_hot_action:
            actions = torch.nn.functional.one_hot(
                actions.long(), num_classes=self.action_size
            ).float()

        return frames, actions


class MazeProvider(TrainBatchProvider):
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(self.loader)

    def next_batch(self, step: int):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class MazeAdapter(EnvironmentAdapter):
    name = "3d_maze"

    def configure_args(self, args):
        args.env_type = "3d_maze"
        args.action_size = ACTION_SIZE
        args.loss_type = "mse"
        args.obs_bit = 5
        args.maze_data_dir = resolve_maze_data_dir(getattr(args, "maze_data_dir", None))
        return args

    def build_context(self, args) -> EnvTrainContext:
        full_seq_len = args.init_size + args.seq_size + 5

        train_dataset = MazeDataset(
            full_seq_len, partition="train", data_dir=args.maze_data_dir
        )
        test_dataset = MazeDataset(
            full_seq_len, partition="test", data_dir=args.maze_data_dir
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

        eval_obs, eval_act = next(iter(test_loader))
        eval_obs = preprocess(eval_obs.to(args.device), args.obs_bit)
        eval_act = eval_act.to(args.device)

        return EnvTrainContext(
            train_provider=MazeProvider(train_loader),
            test_loader=test_loader,
            eval_obs=eval_obs,
            eval_act=eval_act,
            action_size=args.action_size,
            use_amp=args.use_amp and args.device.startswith("cuda"),
            checkpoint_interval=5000,
        )

    def process_batch(self, batch, args):
        obs, act = batch
        return obs.to(args.device), act.to(args.device)

    def log_metrics(self, mode: str, results, writer, step: int, wandb_run=None) -> float:
        metrics: Dict[str, float] = {
            f"{mode}/損失": results["train_loss"].item(),
            f"{mode}/再構成誤差": results["obs_cost"].mean().item(),
            f"{mode}/KL_abs": results["kl_abs_state"].mean().item(),
            f"{mode}/KL_obs": results["kl_obs_state"].mean().item(),
            f"{mode}/KL_mask": results["kl_mask"].mean().item(),
        }
        if "q_mask" in results:
            metrics[f"{mode}/境界確率平均"] = results["q_mask"].mean().item()

        writer.add_scalar(f"{mode}/損失", metrics[f"{mode}/損失"], step)
        writer.add_scalar(f"{mode}/再構成誤差", metrics[f"{mode}/再構成誤差"], step)
        writer.add_scalar(f"{mode}/KL_abs", metrics[f"{mode}/KL_abs"], step)
        writer.add_scalar(f"{mode}/KL_obs", metrics[f"{mode}/KL_obs"], step)
        writer.add_scalar(f"{mode}/KL_mask", metrics[f"{mode}/KL_mask"], step)
        if f"{mode}/境界確率平均" in metrics:
            writer.add_scalar(f"{mode}/境界確率平均", metrics[f"{mode}/境界確率平均"], step)

        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

        return metrics[f"{mode}/損失"]

    def on_best_model(self, model, ctx: EnvTrainContext, args, step: int):
        visualize_results(model, ctx.test_loader, args, seq_idx=0)
        model.train()
