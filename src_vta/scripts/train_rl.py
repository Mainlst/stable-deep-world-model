'''
Dreamerスタイル (VTA) の強化学習ループ
"python train_rl.py --config config.py"
'''
import sys
sys.path.append('../')  # stable-deep-world-model/src_vtaへのパスを追加

import argparse
import os
import time
import random
import numpy as np
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# Gym環境 (必要に応じて pip install gymnasium)
import gymnasium as gym 
# もしくは import gym

# 作成したモジュールをインポート
from config import load_config
from models.vta import VTA
from models.agent import DreamerAgent
from models.policy import VTAPolicy

from envs.config import DefaultEnvConfig
from envs.factory import build_vector_env
from envs.register import register_all
register_all()

# -----------------------------------------------------------------------------
# 1. Experience Replay Buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, observation_shape, action_size):
        self.capacity = capacity
        self.episodes = [] # List of dictionaries
        self.total_steps = 0
        self.obs_shape = observation_shape
        self.act_size = action_size

    def add_episode(self, episode_data):
        """
        エピソード完了時にデータを保存する
        episode_data: {
            "obs": [ (C,H,W), ... ],
            "action": [ (ActDim,), ... ],
            "reward": [ float, ... ],
            "done": [ bool, ... ]
        }
        """
        self.episodes.append(episode_data)
        self.total_steps += len(episode_data["action"])
        
        # 容量オーバー時の古いエピソード削除
        while self.total_steps > self.capacity:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed["action"])

    def sample_batch(self, batch_size, seq_len):
        """
        時系列データをランダムサンプリングする
        Returns:
            obs: (B, T, C, H, W)
            actions: (B, T, ActDim)
            rewards: (B, T, 1)
            dones: (B, T, 1)
        """
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []

        for _ in range(batch_size):
            # ランダムにエピソードを選択
            while True:
                episode = random.choice(self.episodes)
                ep_len = len(episode["action"])
                if ep_len >= seq_len:
                    break
            
            # ランダムな開始位置を選択
            start_idx = random.randint(0, ep_len - seq_len)
            end_idx = start_idx + seq_len

            # データの切り出し
            # obsリストはnumpy arrayのリストを想定
            o = np.stack(episode["obs"][start_idx:end_idx])
            a = np.stack(episode["action"][start_idx:end_idx])
            r = np.array(episode["reward"][start_idx:end_idx], dtype=np.float32)
            d = np.array(episode["done"][start_idx:end_idx], dtype=np.float32)

            obs_batch.append(o)
            act_batch.append(a)
            rew_batch.append(r)
            done_batch.append(d)

        # numpy -> torch tensor
        obs_batch = torch.from_numpy(np.stack(obs_batch)).float()
        act_batch = torch.from_numpy(np.stack(act_batch)).float()
        rew_batch = torch.from_numpy(np.stack(rew_batch)).float().unsqueeze(-1) # (B, T, 1)
        done_batch = torch.from_numpy(np.stack(done_batch)).float().unsqueeze(-1)

        return obs_batch, act_batch, rew_batch, done_batch


# -----------------------------------------------------------------------------
# 3. Main Training Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config: {cfg.exp_name} | Device: {cfg.device}")
    
    # ログ設定
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = cfg.work_dir / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- 環境の構築 ---
    # ※ ここで実際の環境を指定してください (例: 'CartPole-v1', 'MiniGrid', custom env)
    env_cfg = DefaultEnvConfig(
        env_id='MemoryMaze-9x9-v0',
        num_envs=1,
        img_size=(64, 64),
        frame_stack=1,
        action_repeat=1,
    )
    env = build_vector_env(env_cfg, output_dir=str(log_dir), shared_memory=False)
    
    # --- モデル構築 ---
    vta_model = VTA(
        belief_size=cfg.belief_size,
        state_size=cfg.state_size,
        act_size=cfg.action_size,
        num_layers=cfg.num_layers,
        max_seg_len=cfg.seg_len,
        max_seg_num=cfg.seg_num,
        loss_type=cfg.loss_type,
    ).to(cfg.device)

    agent = DreamerAgent(vta_model, cfg).to(cfg.device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learn_rate)
    
    # チェックポイント読み込み
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume)
        agent.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iteration"]

    # --- Replay Buffer & Policy ---
    # 画像サイズ等は環境に合わせてください
    buffer = ReplayBuffer(
        capacity=100000, 
        observation_shape=(3, 64, 64), 
        action_size=cfg.action_size
    )
    
    policy = VTAPolicy(agent, cfg.device)

    # --- 学習設定 ---
    train_every = 5       # 5ステップごとに学習 (環境ステップ数比)
    collect_steps = 1000  # 最初にランダム行動で集めるステップ数 (Seed Episodes)
    batch_seq_len = cfg.init_size + cfg.seq_size # 学習時のシーケンス長 (Init + Train)
    
    print("Start collection & training loop...")
    
    obs, _ = env.reset()
    policy.reset()
    
    # エピソード一時保存用
    current_episode = {"obs": [], "action": [], "reward": [], "done": []}
    
    iteration = start_iter
    total_env_steps = 0
    
    while iteration < cfg.max_iters:
        
        # -------------------------------------------------------
        # A. 行動決定 (Explore or Policy)
        # -------------------------------------------------------
        if total_env_steps < collect_steps:
            # 初期はランダム行動
            if hasattr(env.action_space, 'sample'):
                 action = env.action_space.sample()
            else:
                 action = np.random.randn(cfg.action_size) # Dummy
        else:
            # Policyによる行動
            action = policy(obs, eval_mode=False)

        # -------------------------------------------------------
        # B. 環境ステップ
        # -------------------------------------------------------
        next_obs, reward, done, truncated, _ = env.step(action)
        
        # バッファへの保存用にデータを整形
        current_episode["obs"].append(obs)
        current_episode["action"].append(action)
        current_episode["reward"].append(reward)
        current_episode["done"].append(done) # 0.0 or 1.0
        
        obs = next_obs
        total_env_steps += 1
        
        # エピソード終了時の処理
        if done or truncated:
            # エピソードをバッファに追加
            buffer.add_episode(current_episode)
            
            # リセット
            obs, _ = env.reset()
            policy.reset()
            current_episode = {"obs": [], "action": [], "reward": [], "done": []}
            
            print(f"Episode finished. Buffer size: {buffer.total_steps} steps")

        # -------------------------------------------------------
        # C. 学習ステップ (Training)
        # -------------------------------------------------------
        # バッファに十分なデータがあり、かつ学習タイミングであれば学習実行
        should_train = (
            total_env_steps >= collect_steps and 
            total_env_steps % train_every == 0 and
            buffer.total_steps > batch_seq_len * cfg.batch_size
        )

        if should_train:
            agent.train()
            
            # バッファからサンプリング
            # (B, T, C, H, W), (B, T, Act), ...
            b_obs, b_act, b_rew, b_done = buffer.sample_batch(
                cfg.batch_size, batch_seq_len
            )
            
            b_obs = b_obs.to(cfg.device)
            b_act = b_act.to(cfg.device)
            b_rew = b_rew.to(cfg.device)
            b_done = b_done.to(cfg.device)

            # 勾配リセット
            optimizer.zero_grad()

            # Forward & Loss
            total_loss, logs = agent(
                b_obs, b_act, b_rew, b_done, 
                seq_len=cfg.seq_size, 
                init_len=cfg.init_size
            )

            # Backward
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
            optimizer.step()

            iteration += 1

            # ログ出力
            if iteration % 10 == 0:
                print(f"Iter {iteration} (EnvSteps {total_env_steps}) | "
                      f"Total Loss: {logs['train_loss']:.4f} | "
                      f"VTA: {logs['vta_loss']:.4f} | "
                      f"Actor: {logs['actor_loss']:.4f} | "
                      f"Value: {logs['value_loss']:.4f}")

            # 保存
            if iteration % 1000 == 0:
                save_path = log_dir / f"checkpoint_{iteration}.pth"
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg.__dict__
                }, save_path)
                print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()