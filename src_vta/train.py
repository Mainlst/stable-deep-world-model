'''
Bouncing Balls環境用の学習スクリプト
'''

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc               # メモリ管理用
import matplotlib       # バックエンド設定用
import shutil           # ファイル操作用

# GUIのない環境でのクラッシュを防ぐ設定
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 自作モジュールのインポート
from config import Config
from model import VTA
from utils import visualize_results, preprocess

def main():
    # ----------------------------------------------------
    # 1. 設定と準備
    # ----------------------------------------------------
    args = Config()
    print(f"Device: {args.device}")
    print(f"Loss Type: {args.loss_type}")
    
    # シード固定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    if args.env_type == "3d_maze":
        from maze_env import generate_vta_dataset
    else:
        from bouncing_balls import generate_vta_dataset
    
    # ログディレクトリのクリーンアップと作成
    log_dir = args.work_dir / args.exp_name / "logs"
    if log_dir.exists():
        # 過去のログが混ざらないように一旦削除（任意）
        # shutil.rmtree(log_dir) 
        pass
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # モデル保存ディレクトリ
    save_dir = args.work_dir / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # データ生成パラメータ
    # init_size(5) + seq_size(20) + 予備(5) = 30フレーム程度
    SEQ_LEN_GEN = args.init_size + args.seq_size + 5
    
    # ----------------------------------------------------
    # 2. テストデータ (固定・評価用)
    # ----------------------------------------------------
    print("Generating FIXED test data (500 seqs)...")
    test_data_raw = generate_vta_dataset(500, seq_len=SEQ_LEN_GEN, size=32, dt=args.dt)
    test_loader = DataLoader(test_data_raw, batch_size=args.batch_size, shuffle=False)
    
    # テスト用バッチの確保（可視化用）
    pre_test_full_data_list = next(iter(test_loader))[0].to(args.device)
    pre_test_full_data_list = preprocess(pre_test_full_data_list, args.obs_bit)
    action_size = getattr(args, 'action_size', 0)
    test_act_list = torch.zeros(
        pre_test_full_data_list.size(0), 
        pre_test_full_data_list.size(1), 
        action_size
    ).to(args.device)

    # ----------------------------------------------------
    # 3. 学習データの運用戦略 (チャンク分割)
    # ----------------------------------------------------
    # メモリ枯渇を防ぐため、全学習期間を N分割 し、その都度データを生成する
    NUM_CHUNKS = 1  # 4回作り直す (学習の1/4ごとにリフレッシュ)
    REFRESH_STEPS = args.max_iters // NUM_CHUNKS
    
    # 1回あたりのデータ生成量 (50,000の1/4 = 12,500系列ならメモリ16GBでも余裕)
    # configのepoch_data_sizeに関わらず、ここで安全な値を指定します
    CHUNK_DATA_SIZE =  args.epoch_data_size // NUM_CHUNKS
    
    print(f"Training Strategy: Refresh data every {REFRESH_STEPS} steps.")
    print(f"Chunk Size: {CHUNK_DATA_SIZE} sequences.")

    # 最初のデータチャンクを生成
    print("Generating Initial Data Chunk...")
    current_data = generate_vta_dataset(CHUNK_DATA_SIZE, seq_len=SEQ_LEN_GEN, size=32, dt=args.dt)
    train_loader = DataLoader(current_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_iter = iter(train_loader) # 無限ループ用のイテレータ

    # ----------------------------------------------------
    # 4. モデルと最適化
    # ----------------------------------------------------
    model = VTA(
        belief_size=args.belief_size,
        state_size=args.state_size,
        act_size=action_size,
        num_layers=args.num_layers,
        max_seg_len=args.seg_len,
        max_seg_num=args.seg_num,
        loss_type=args.loss_type 
    ).to(args.device)
    
    # ★重要: 前回の議論にあった「初期バイアス修正」を model.py に適用していない場合、
    # ここで無理やり適用することも可能です（推奨は model.py の修正）
    # model.state_model.prior_boundary.network.bias.data = torch.tensor([1.0, -1.0]).to(args.device)
    # model.state_model.post_boundary.network[-1].bias.data = torch.tensor([1.0, -1.0]).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, amsgrad=True)

    # ----------------------------------------------------
    # 5. 学習ループ
    # ----------------------------------------------------
    model.train()
    b_idx = 0
    best_val_loss = float('inf')
    
    pbar = tqdm(total=args.max_iters)

    while b_idx < args.max_iters:
        # (A) データの取り出し -------------------------
        try:
            batch = next(train_iter)
        except StopIteration:
            # チャンク内のデータを一周しきったら、同じデータでもう一度イテレータを作る
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        b_idx += 1

        # (B) Gumbel-Softmax アニーリング ----------------
        if args.beta_anneal:
            # 指数関数的減衰: beta_anneal=75 程度推奨
            current_beta = (args.max_beta - args.min_beta) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
            model.state_model.mask_beta = current_beta
        else:
            model.state_model.mask_beta = args.max_beta

        # (C) 前処理と転送 -----------------------------
        obs_raw = batch[0].to(args.device)
        train_obs_list = preprocess(obs_raw, args.obs_bit)
        train_act_list = torch.zeros(
            train_obs_list.size(0), 
            train_obs_list.size(1), 
            action_size
        ).to(args.device)

        # (D) 更新ステップ -----------------------------
        optimizer.zero_grad()
        results = model(
            train_obs_list,
            train_act_list,
            args.seq_size,
            args.init_size,
            args.obs_std,
            loss_type=args.loss_type
        )
        
        # 境界KL項の重み付け (必要に応じて調整。現状は1.0)
        kl_mask_loss = results['kl_mask'].mean()
        # loss = results['train_loss'] # デフォルト
        
        # もし境界過多が直らない場合、ここを少し強める (例: + 5.0 * kl_mask_loss)
        # ただし、results['train_loss']には既に1.0倍が含まれているので注意
        loss = results['train_loss'] 

        loss.backward()
        
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # (E) ログ記録 --------------------------------
        if b_idx % 10 == 0:
            pbar.set_description(f"L:{loss.item():.2f}|B:{model.state_model.mask_beta:.3f}")
            pbar.update(10)
            writer.add_scalar("Train/Loss", loss.item(), b_idx)
            writer.add_scalar("Train/Beta", model.state_model.mask_beta, b_idx)
            
            # 境界確率の平均 (0に近いほどCOPY, 1に近いほどUPDATE)
            if 'q_mask' in results:
                writer.add_scalar("Train/Q_Mask_Mean", results['q_mask'].mean().item(), b_idx)

        # (F) データの再生成 (チャンク更新) -------------
        if b_idx % REFRESH_STEPS == 0 and b_idx < args.max_iters:
            tqdm.write(f"[{b_idx}] Refreshing Data Chunk (Memory Cleanup)...")
            
            # 変数削除
            del batch
            del train_obs_list
            del results
            del loss
            del train_iter
            del train_loader
            del current_data
            
            # 強制GC (これがSegfaultを防ぐ鍵)
            gc.collect()
            torch.cuda.empty_cache() # GPUメモリも整理
            
            # 新しいデータを生成
            current_data = generate_vta_dataset(CHUNK_DATA_SIZE, seq_len=SEQ_LEN_GEN, size=32, dt=args.dt)
            train_loader = DataLoader(current_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
            train_iter = iter(train_loader)

        # (G) 定期保存 (1万ステップごと) ----------------
        if b_idx % 10000 == 0:
            ckpt_path = save_dir / f"ckpt_{b_idx}.pt"
            torch.save(model.state_dict(), ckpt_path)
            # tqdm.write(f"Saved checkpoint: {ckpt_path}")

        # (H) 評価とベストモデル保存 --------------------
        if b_idx % 500 == 0: # 頻度はお好みで
            with torch.no_grad():
                model.eval()
                val_results = model(
                    pre_test_full_data_list,
                    test_act_list,
                    args.seq_size,
                    args.init_size,
                    args.obs_std,
                    loss_type=args.loss_type
                )
                val_loss = val_results['train_loss'].item()
                writer.add_scalar("Test/Loss", val_loss, b_idx)
                
                model.train() # 忘れずに戻す

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_dir / "model_best.pt")
                    # 可視化画像の保存
                    visualize_results(model, test_loader, args, seq_idx=0)
                    model.train()

    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    main()