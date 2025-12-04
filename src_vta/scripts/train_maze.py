'''
3D_Maze環境用の学習スクリプト
'''

import argparse
import glob
import sys
import gc               # メモリ管理用
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 自作モジュールのインポート
from src_vta.config import load_config
from src_vta.models import VTA
from src_vta.utils import preprocess, visualize_results

# -----------------------------------------------------------------------------
# Dataset Class (ユーザー提供のコード)
# -----------------------------------------------------------------------------
ACTION_SIZE = 3

class MazeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        length,
        partition="train",
        image_width=32,
        image_height=32,
        image_channels=3,
        one_hot_action=True,
    ):
        self.path = "3d_maze_default"
        self.partition = partition
        self.length = length
        self.height = image_height
        self.width = image_width
        self.image_channels = image_channels
        self.one_hot_action = one_hot_action
        self.args = Config()

        # self._load_data()
        # データをロードせず、ファイルパスのリストだけを作る
        dir_path = f"{self.path}/{self.partition}"
        self.file_paths = glob.glob(f"{dir_path}/*.npz")
        
        if len(self.file_paths) == 0:
            print(f"Error: No .npz files found in {dir_path}")
            sys.exit(1)

        print(f"Dataset ({self.partition}): {len(self.file_paths)} episodes found (Lazy Loading).")

    def _load_data(self) -> None:
        dir_path = f"{self.path}/{self.partition}"
        self.file_paths = glob.glob(f"{dir_path}/*.npz")
        
        # データがない場合の例外処理
        if len(self.file_paths) == 0:
            print(f"Error: No .npz files found in {dir_path}")
            print("Please run generate_npz.py first.")
            sys.exit(1)

        self.data = {"video": [], "actions": []}
        for path in tqdm(self.file_paths, desc=f"Loading {self.partition} samples"):
            sample_episode = np.load(path, allow_pickle=True)
            frames = self._resize(sample_episode["video"])
            self.data["video"].append(frames)
            self.data["actions"].append(sample_episode["actions"])

        self.data["video"] = np.stack(self.data["video"], axis=0)
        self.data["actions"] = np.stack(self.data["actions"], axis=0)

        print(f"Dataset ({self.partition}): {self.data['video'].shape[0]} episodes, "
              f"videos: {self.data['video'].shape}, actions: {self.data['actions'].shape}")

    def _resize(self, frames: np.ndarray):
        return np.stack(
            [cv2.resize(frame, (self.width, self.height)) for frame in frames]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # ★変更点: データが必要になった瞬間にファイルから読み込む
        path = self.file_paths[index]
        
        # コンテキストマネージャ(with)を使って安全にロード
        with np.load(path, allow_pickle=True) as sample_episode:
            frames_raw = sample_episode["video"]
            actions_raw = sample_episode["actions"]
            
        # 以降は元の処理と同じ（リサイズ等は都度行う）
        frames = self._resize(frames_raw)
        
        frames = torch.from_numpy(frames)
        actions = torch.from_numpy(actions_raw)

        seq_len = frames.shape[0]
        if seq_len > self.length:
            start_t = np.random.randint(0, seq_len - self.length)
        else:
            start_t = 0

        frames = frames[start_t:start_t + self.length]
        actions = actions[start_t:start_t + self.length]

        frames = frames.permute(0, 3, 1, 2)
        frames = frames.float() / 255.0

        if self.one_hot_action:
            actions = torch.nn.functional.one_hot(
                actions.long(), num_classes=self.args.action_size
            ).float()

        return frames, actions

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def log(mode, results, writer, step):
    """Tensorboardへのログ記録"""
    # 損失の記録
    writer.add_scalar(f"{mode}/Loss", results['train_loss'].item(), step)
    writer.add_scalar(f"{mode}/Obs_Cost", results['obs_cost'].mean().item(), step)
    writer.add_scalar(f"{mode}/KL_Abs", results['kl_abs_state'].mean().item(), step)
    writer.add_scalar(f"{mode}/KL_Obs", results['kl_obs_state'].mean().item(), step)
    writer.add_scalar(f"{mode}/KL_Mask", results['kl_mask'].mean().item(), step)
    
    # 境界確率の記録 (0=COPY, 1=UPDATE)
    if 'q_mask' in results:
        writer.add_scalar(f"{mode}/Q_Mask_Mean", results['q_mask'].mean().item(), step)
    
    return results['train_loss'].item()

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train VTA on 3D Maze dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    cli_args = parser.parse_args()
    args = load_config(cli_args.config, exp_name=cli_args.exp_name)
    
    # Maze用の強制設定 (Configクラスで設定済みなら不要だが念のため)
    args.env_type = "3d_maze"
    args.action_size = ACTION_SIZE
    args.loss_type = "mse" # 連続値画像なのでMSE
    args.obs_bit = 5       # 5bit量子化 (論文準拠)
    
    print(f"Device: {args.device}")
    print(f"Action Size: {args.action_size}")
    
    # ログディレクトリ設定
    log_dir = args.work_dir / args.exp_name / "logs"
    writer = SummaryWriter(str(log_dir))
    save_dir = args.work_dir / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. データローダーの準備
    # Datasetクラスは __getitem__ で length 分のデータを返す
    # 必要な長さ = init_size (コンテキスト) + seq_size (学習対象)
    full_seq_len = args.init_size + args.seq_size + 5  # 予備5フレーム分も確保
    
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

    # 3. モデル構築
    model = VTA(
        belief_size=args.belief_size,
        state_size=args.state_size,
        act_size=args.action_size,
        num_layers=args.num_layers,
        max_seg_len=args.seg_len,
        max_seg_num=args.seg_num,
        loss_type=args.loss_type
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, amsgrad=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and args.device.startswith("cuda"))

    # テスト用の固定バッチ（学習中の定点観測用）
    # Datasetの仕様に合わせて (frames, actions) を受け取る
    pre_test_full_data_list, test_act_list = next(iter(test_loader))
    pre_test_full_data_list = pre_test_full_data_list.to(args.device)
    test_act_list = test_act_list.to(args.device)
    
    # 前処理（量子化など）
    pre_test_full_data_list = preprocess(pre_test_full_data_list, args.obs_bit)

    # 学習変数の初期化
    b_idx = 0
    best_val_loss = float('inf') # 損失は低い方が良いので無限大で初期化
    
    print("Start Training...")
    
    # 無限ループではなくmax_itersまで回す構造にするため、epochループで構成
    # tqdmで進捗表示
    pbar = tqdm(total=args.max_iters)
    
    while b_idx < args.max_iters:
        # -------------
        #  訓練ループ
        # -------------
        for train_obs_list, train_act_list in train_loader:
            if b_idx >= args.max_iters:
                break
                
            b_idx += 1
            
            # Gumbel-Softmaxの温度パラメータのアニーリング
            if args.beta_anneal:
                # 指数減衰
                model.state_model.mask_beta = (args.max_beta - args.min_beta) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
            else:
                model.state_model.mask_beta = args.max_beta
            
            # データの転送と前処理
            train_obs_list = train_obs_list.to(args.device)
            train_act_list = train_act_list.to(args.device)
            train_obs_list = preprocess(train_obs_list, args.obs_bit)

            # 順伝播
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp and args.device.startswith("cuda")):
                results = model(
                    train_obs_list,
                    train_act_list,
                    args.seq_size,
                    args.init_size,
                    obs_std=args.obs_std,
                    loss_type=args.loss_type
                )

            # 逆伝播・パラメータ更新
            train_total_loss = results['train_loss']
            scaler.scale(train_total_loss).backward()
            
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # プログレスバー更新
            if b_idx % 10 == 0:
                log( "train", results, writer, b_idx)
                pbar.set_description(f"Loss: {train_total_loss.item():.2f} | Beta: {model.state_model.mask_beta:.3f}")
                pbar.update(10)

            # ----------------------
            #  テストバッチで評価
            # ----------------------
            if b_idx % 500 == 0: # 頻度は適宜調整 (例: 100 or 500)
                with torch.no_grad():
                    model.eval()
                    autocast_enabled = args.use_amp and args.device.startswith("cuda")
                    with torch.cuda.amp.autocast(enabled=autocast_enabled):
                        val_results = model(
                            pre_test_full_data_list,
                            test_act_list,
                            args.seq_size,
                            args.init_size,
                            obs_std=args.obs_std,
                            loss_type=args.loss_type
                        )

                    # ログ記録
                    val_loss = log("test", val_results, writer, b_idx)

                    # モデル保存 (損失が下がったら更新)
                    # 元のコードは > best_val_loss でしたが、損失は最小化問題なので < を使用します
                    if val_loss < best_val_loss:
                        tqdm.write(f"[{b_idx}] New best loss: {val_loss:.4f}. Saving model.")
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), str(save_dir / "model_best.pt"))
                        
                        # ベストモデル更新時に可視化画像を保存
                        visualize_results(model, test_loader, args, seq_idx=0)
                    
                    # 定期チェックポイント保存
                    if b_idx % 5000 == 0:
                        torch.save(model.state_dict(), str(save_dir / f"ckpt_{b_idx}.pt"))
                        
                    # ★追加: 評価と可視化が終わったタイミングでメモリ掃除
                    gc.collect()
                    torch.cuda.empty_cache() # GPUメモリも解放（もしGPU使用なら）

    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    main()
