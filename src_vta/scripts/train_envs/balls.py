import gc
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_vta.data.bouncing_balls import generate_vta_dataset
from src_vta.utils import preprocess, visualize_results
from .base import EnvironmentAdapter, EnvTrainContext, TrainBatchProvider


class BouncingBallsProvider(TrainBatchProvider):
    """生成データをチャンクごとに差し替えてメモリ消費を抑える。"""

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
            tqdm.write(f"[{step}] データチャンクを再生成しています（Bouncing Balls）")
            self._refresh()

        return batch


class BouncingBallsAdapter(EnvironmentAdapter):
    name = "bouncing_balls"

    def configure_args(self, args):
        args.env_type = "bouncing_balls"
        args.action_size = 0
        args.loss_type = "bce"
        args.obs_bit = getattr(args, "obs_bit", None)
        return args

    def build_context(self, args) -> EnvTrainContext:
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

        return EnvTrainContext(
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
        metrics: Dict[str, float] = {
            f"{mode}/損失": loss_val,
            f"{mode}/β": results.get("beta", 0.0) if isinstance(results, dict) else 0.0,
        }
        if "q_mask" in results:
            metrics[f"{mode}/境界確率平均"] = results["q_mask"].mean().item()

        writer.add_scalar(f"{mode}/損失", loss_val, step)
        writer.add_scalar(f"{mode}/β", metrics[f"{mode}/β"], step)
        if f"{mode}/境界確率平均" in metrics:
            writer.add_scalar(f"{mode}/境界確率平均", metrics[f"{mode}/境界確率平均"], step)

        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

        return loss_val

    def on_best_model(self, model, ctx: EnvTrainContext, args, step: int):
        visualize_results(model, ctx.test_loader, args, seq_idx=0)
        model.train()
