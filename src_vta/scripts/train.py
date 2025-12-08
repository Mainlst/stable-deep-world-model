"""
統合トレーナーのエントリポイント。
環境ごとの学習ロジックは train_envs/ 以下の各アダプターに分離しています。
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from src_vta.config import load_config
from src_vta.models import VTA
from src_vta.utils import config_to_dict, preprocess
from src_vta.scripts.train_envs import ADAPTERS, EnvTrainContext, EnvironmentAdapter


def _set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_training(args, adapter: EnvironmentAdapter):
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
            adapter.log_metrics("学習", results, writer, b_idx, wandb_run)
            pbar.set_description(
                f"損失:{loss.item():.2f}|β:{model.state_model.mask_beta:.3f}"
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
                val_loss = adapter.log_metrics("評価", val_results, writer, b_idx, wandb_run)
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_dir / "model_best.pt")
                    adapter.on_best_model(model, ctx, args, b_idx)

                if ctx.checkpoint_interval and b_idx % ctx.checkpoint_interval == 0:
                    torch.save(model.state_dict(), save_dir / f"ckpt_{b_idx}.pt")

    writer.close()
    wandb_run.finish()
    print("学習が完了しました。")


def build_parser(default_env: Optional[str] = None):
    parser = argparse.ArgumentParser(description="Unified VTA trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    parser.add_argument(
        "--env",
        type=str,
        default=default_env,
        choices=list(ADAPTERS.keys()),
        help="学習対象の環境 (例: bouncing_balls, 3d_maze)",
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
        raise ValueError(f"未対応の環境です: {cfg.env_type}")

    run_training(cfg, adapter)


if __name__ == "__main__":
    main()
