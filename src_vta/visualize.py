'''
学習済みのVTAモデルを用いて，指定された環境のシーケンスを可視化するスクリプト
'''

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 既存のモジュール
from config import Config
from model import VTA
from utils import visualize_results

# ★変更点: 環境に応じてインポートを切り替えるため、ここでの固定インポートは削除
# from bouncing_balls import generate_vta_dataset 

def main():
    # -------------------------------------------------
    # 1. 引数の設定
    # -------------------------------------------------
    parser = argparse.ArgumentParser(description="VTAの学習済みモデルを可視化するスクリプト")
    parser.add_argument('ckpt_path', type=str, help='読み込むチェックポイントファイルのパス (.pt)')
    parser.add_argument('--idx', type=int, default=0, help='可視化するバッチ内のインデックス (デフォルト: 0)')
    parser.add_argument('--num_samples', type=int, default=10, help='生成するテストデータの数')
    args = parser.parse_args()

    # -------------------------------------------------
    # 2. 設定の読み込みとデータセット関数のインポート
    # -------------------------------------------------
    config = Config()
    device = config.device
    print(f"Device: {device}")
    print(f"Environment: {config.env_type}") # 確認用

    # ★変更点: 環境に合わせてデータセット生成関数を読み込む
    if config.env_type == "3d_maze":
        from maze_env import generate_vta_dataset
    else:
        from bouncing_balls import generate_vta_dataset

    # チェックポイントのパス確認
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        sys.exit(1)

    # -------------------------------------------------
    # 3. モデルの構築と重みのロード
    # -------------------------------------------------
    print("Building model...")
    model = VTA(
        belief_size=config.belief_size,
        state_size=config.state_size,
        act_size=config.action_size,
        num_layers=config.num_layers,
        max_seg_len=config.seg_len,
        max_seg_num=config.seg_num,
        loss_type=config.loss_type 
    ).to(device)

    print(f"Loading weights from {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and keys_match(model, checkpoint):
             model.load_state_dict(checkpoint)
        else:
             model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # -------------------------------------------------
    # 4. テストデータの生成
    # -------------------------------------------------
    print("Generating test data...")
    SEQ_LEN_GEN = config.init_size + config.seq_size + 5
    
    # Maze環境の場合、dtは無視されますが引数として渡しても問題ありません
    test_data = generate_vta_dataset(max(args.num_samples, args.idx + 1), seq_len=SEQ_LEN_GEN, size=32)
    test_loader = DataLoader(test_data, batch_size=args.num_samples, shuffle=False)

    # -------------------------------------------------
    # 5. 可視化の実行
    # -------------------------------------------------
    print(f"Visualizing sequence index: {args.idx}")
    visualize_results(model, test_loader, config, seq_idx=args.idx)
    
    print(f"Done. Check the output image in: {config.work_dir}/vis_seq_{args.idx}.png")

def keys_match(model, state_dict):
    model_keys = set(model.state_dict().keys())
    dict_keys = set(state_dict.keys())
    return len(model_keys.intersection(dict_keys)) > 0

if __name__ == "__main__":
    main()
