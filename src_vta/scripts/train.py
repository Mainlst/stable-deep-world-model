"""
Unified training entrypoint for VTA across supported environments.

Environment specific pieces are provided by adapters so that adding a new
environment only requires implementing dataset + logging glue, while the
core training loop stays shared.
"""

from __future__ import annotations

import argparse
import gc
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from src_vta.config import load_config
from src_vta.data.bouncing_balls import generate_vta_dataset
from src_vta.models import VTA
from src_vta.utils import config_to_dict, preprocess, visualize_results

# -----------------------------------------------------------------------------#
# Types
# -----------------------------------------------------------------------------#


@dataclass
class TrainContext:
    """Runtime objects the shared trainer needs."""

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
    """Provides batches to the trainer while hiding how data is refreshed."""

    def next_batch(self, step: int):
        raise NotImplementedError


class TrainingAdapter:
    """Environment specific hooks."""

    name: str

    def configure_args(self, args):
        return args

    def build_context(self, args) -> TrainContext:
        raise NotImplementedError

    def process_batch(self, batch, args) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def log_metrics(self, mode: str, results, writer, step: int, wandb_run=None) -> float:
        raise NotImplementedError

    def on_best_model(self, model: VTA, ctx: TrainContext, args, step: int):
        visualize_results(model, ctx.test_loader, args, seq_idx=0)
        model.train()


# -----------------------------------------------------------------------------#
# Bouncing Balls adapter
# -----------------------------------------------------------------------------#


class BouncingBallsProvider(TrainBatchProvider):
    """Regenerates synthetic data chunks to keep memory use low."""

    def __init__(self, args, chunk_data_size: int, refresh_steps: int, seq_len: int):
        self.args = args
        self.chunk_data_size = chunk_data_size
        self.refresh_steps = max(refresh_steps, 1)
        self.seq_len = seq_len
        self._build_loader()

    def _build_loader(self):
        self.current_data = generate_vta_dataset(
            self.chunk_data_size, seq_len=self.seq_len, size=32, dt=self.args.dt
        )
        self.loader = DataLoader(
            self.current_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )
        self.iterator = iter(self.loader)

    def _refresh(self):
        del self.iterator
        del self.loader
        del self.current_data
        gc.collect()
        torch.cuda.empty_cache()
        self._build_loader()

    def next_batch(self, step: int):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)

        if step % self.refresh_steps == 0 and step < self.args.max_iters:
            tqdm.write(f"[{step}] Refreshing data chunk (Bouncing Balls).")
            self._refresh()

        return batch


class BouncingBallsAdapter(TrainingAdapter):
    name = "bouncing_balls"

    def configure_args(self, args):
        args.env_type = "bouncing_balls"
        args.action_size = 0
        args.loss_type = "bce"
        args.obs_bit = getattr(args, "obs_bit", None)
        return args

    def build_context(self, args) -> TrainContext:
        seq_len = args.init_size + args.seq_size + 5
        num_chunks = 1
        refresh_steps = args.max_iters // num_chunks
        chunk_data_size = args.epoch_data_size // num_chunks

        provider = BouncingBallsProvider(args, chunk_data_size, refresh_steps, seq_len)

        test_data_raw = generate_vta_dataset(500, seq_len=seq_len, size=32, dt=args.dt)
        test_loader = DataLoader(
            test_data_raw,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            drop_last=False,
        )
        eval_obs = preprocess(next(iter(test_loader))[0].to(args.device), args.obs_bit)
        eval_act = torch.zeros(
            eval_obs.size(0), eval_obs.size(1), args.action_size, device=args.device
        )

        return TrainContext(
            train_provider=provider,
            test_loader=test_loader,
            eval_obs=eval_obs,
            eval_act=eval_act,
            action_size=0,
            use_amp=False,
            checkpoint_interval=10000,
        )

    def process_batch(self, batch, args):
        obs_raw = batch[0].to(args.device)
        act = torch.zeros(
            obs_raw.size(0), obs_raw.size(1), args.action_size, device=args.device
        )
        return obs_raw, act

    def log_metrics(self, mode: str, results, writer, step: int, wandb_run=None) -> float:
        loss_val = results["train_loss"].item()
        metrics = {
            f"{mode}/loss": loss_val,
            f"{mode}/beta": results.get("beta", 0.0)
            if isinstance(results, dict)
            else 0.0,
        }
        if "q_mask" in results:
            metrics[f"{mode}/q_mask_mean"] = results["q_mask"].mean().item()

        writer.add_scalar(f"{mode}/Loss", metrics[f"{mode}/loss"], step)
        writer.add_scalar(f"{mode}/Beta", metrics[f"{mode}/beta"], step)
        if f"{mode}/q_mask_mean" in metrics:
            writer.add_scalar(
                f"{mode}/Q_Mask_Mean", metrics[f"{mode}/q_mask_mean"], step
            )

        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

        return loss_val


# -----------------------------------------------------------------------------#
# 3D Maze adapter
# -----------------------------------------------------------------------------#


ACTION_SIZE = 3


class MazeDataset(torch.utils.data.Dataset):
    """Lazy loading dataset for 3D Maze npz files."""

    def __init__(
        self,
        length: int,
        partition: str = "train",
        image_width: int = 32,
        image_height: int = 32,
        image_channels: int = 3,
        one_hot_action: bool = True,
    ):
        self.path = "3d_maze_default"
        self.partition = partition
        self.length = length
        self.height = image_height
        self.width = image_width
        self.image_channels = image_channels
        self.one_hot_action = one_hot_action
        self.action_size = ACTION_SIZE

        dir_path = f"{self.path}/{self.partition}"
        self.file_paths = glob.glob(f"{dir_path}/*.npz")

        if len(self.file_paths) == 0:
            print(f"Error: No .npz files found in {dir_path}")
            sys.exit(1)

        print(
            f"Dataset ({self.partition}): {len(self.file_paths)} episodes found (Lazy Loading)."
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


class MazeAdapter(TrainingAdapter):
    name = "3d_maze"

    def configure_args(self, args):
        args.env_type = "3d_maze"
        args.action_size = ACTION_SIZE
        args.loss_type = "mse"
        args.obs_bit = 5
        return args

    def build_context(self, args) -> TrainContext:
        full_seq_len = args.init_size + args.seq_size + 5

        train_dataset = MazeDataset(full_seq_len, partition="train")
        test_dataset = MazeDataset(full_seq_len, partition="test")

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

        return TrainContext(
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
        metrics = {
            f"{mode}/loss": results["train_loss"].item(),
            f"{mode}/obs_cost": results["obs_cost"].mean().item(),
            f"{mode}/kl_abs": results["kl_abs_state"].mean().item(),
            f"{mode}/kl_obs": results["kl_obs_state"].mean().item(),
            f"{mode}/kl_mask": results["kl_mask"].mean().item(),
        }
        if "q_mask" in results:
            metrics[f"{mode}/q_mask_mean"] = results["q_mask"].mean().item()

        writer.add_scalar(f"{mode}/Loss", metrics[f"{mode}/loss"], step)
        writer.add_scalar(f"{mode}/Obs_Cost", metrics[f"{mode}/obs_cost"], step)
        writer.add_scalar(f"{mode}/KL_Abs", metrics[f"{mode}/kl_abs"], step)
        writer.add_scalar(f"{mode}/KL_Obs", metrics[f"{mode}/kl_obs"], step)
        writer.add_scalar(f"{mode}/KL_Mask", metrics[f"{mode}/kl_mask"], step)
        if f"{mode}/q_mask_mean" in metrics:
            writer.add_scalar(
                f"{mode}/Q_Mask_Mean", metrics[f"{mode}/q_mask_mean"], step
            )

        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

        return metrics[f"{mode}/loss"]


# -----------------------------------------------------------------------------#
# Shared trainer
# -----------------------------------------------------------------------------#


ADAPTERS: Dict[str, TrainingAdapter] = {
    BouncingBallsAdapter.name: BouncingBallsAdapter(),
    MazeAdapter.name: MazeAdapter(),
}


def _set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_training(args, adapter: TrainingAdapter):
    args = adapter.configure_args(args)
    _set_seeds(args.seed)

    wandb_run = wandb.init(
        project="stable-deep-world-model",
        name=args.exp_name,
        config=config_to_dict(args),
        dir=str(args.work_dir),
    )

    log_dir = args.work_dir / args.exp_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    save_dir = args.work_dir / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ctx = adapter.build_context(args)

    model = VTA(
        belief_size=args.belief_size,
        state_size=args.state_size,
        act_size=ctx.action_size,
        num_layers=args.num_layers,
        max_seg_len=args.seg_len,
        max_seg_num=args.seg_num,
        loss_type=args.loss_type,
    ).to(args.device)
    wandb_run.watch(model, log="all", log_freq=200)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, amsgrad=True)
    scaler = GradScaler(enabled=ctx.use_amp)

    b_idx = 0
    best_val_loss = float("inf")
    pbar = tqdm(total=args.max_iters)

    while b_idx < args.max_iters:
        b_idx += 1
        batch = ctx.train_provider.next_batch(b_idx)
        obs_raw, act_raw = adapter.process_batch(batch, args)
        obs = preprocess(obs_raw, args.obs_bit)
        act = act_raw

        if args.beta_anneal:
            model.state_model.mask_beta = (
                (args.max_beta - args.min_beta)
                * 0.999 ** (b_idx / args.beta_anneal)
                + args.min_beta
            )
        else:
            model.state_model.mask_beta = args.max_beta

        optimizer.zero_grad()
        autocast_enabled = ctx.use_amp and args.device.startswith("cuda")
        with autocast(enabled=autocast_enabled):
            results = model(
                obs,
                act,
                args.seq_size,
                args.init_size,
                args.obs_std,
                loss_type=args.loss_type,
            )
            loss = results["train_loss"]

        results["beta"] = float(model.state_model.mask_beta)

        if autocast_enabled:
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if b_idx % ctx.log_interval == 0:
            adapter.log_metrics("train", results, writer, b_idx, wandb_run)
            pbar.set_description(
                f"L:{loss.item():.2f}|B:{model.state_model.mask_beta:.3f}"
            )
            pbar.update(ctx.log_interval)

        if b_idx % ctx.eval_interval == 0:
            with torch.no_grad():
                model.eval()
                with autocast(enabled=autocast_enabled):
                    val_results = model(
                        ctx.eval_obs,
                        ctx.eval_act,
                        args.seq_size,
                        args.init_size,
                        args.obs_std,
                        loss_type=args.loss_type,
                    )
                val_results["beta"] = float(model.state_model.mask_beta)
                val_loss = adapter.log_metrics("test", val_results, writer, b_idx, wandb_run)
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_dir / "model_best.pt")
                    adapter.on_best_model(model, ctx, args, b_idx)

                if ctx.checkpoint_interval and b_idx % ctx.checkpoint_interval == 0:
                    torch.save(model.state_dict(), save_dir / f"ckpt_{b_idx}.pt")

    writer.close()
    wandb_run.finish()
    print("Training Finished.")


def build_parser(default_env: Optional[str] = None):
    parser = argparse.ArgumentParser(description="Unified VTA trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    parser.add_argument(
        "--env",
        type=str,
        default=default_env,
        choices=list(ADAPTERS.keys()),
        help="Environment to train on",
    )
    return parser


def main(default_env: Optional[str] = None):
    parser = build_parser(default_env)
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config, exp_name=cli_args.exp_name)
    if cli_args.env:
        cfg.env_type = cli_args.env

    adapter = ADAPTERS.get(cfg.env_type)
    if adapter is None:
        raise ValueError(f"Unsupported env_type: {cfg.env_type}")

    run_training(cfg, adapter)


if __name__ == "__main__":
    main()
