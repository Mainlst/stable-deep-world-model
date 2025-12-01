import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
import matplotlib
# サーバー等でGUIがない場合のエラー回避
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 既存のモジュールをインポート
from config import Config
from model import VTA
from bouncing_balls import generate_vta_dataset
from utils import visualize_results

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
    # 2. 設定の読み込み
    # -------------------------------------------------
    config = Config()
    device = config.device
    print(f"Device: {device}")

    # チェックポイントのパス確認
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        sys.exit(1)

    # -------------------------------------------------
    # 3. モデルの構築と重みのロード
    # -------------------------------------------------
    print("Building model...")
    # configの内容に基づいてモデルを初期化
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
        # map_locationを使うことで、GPUで学習したモデルをCPUのみの環境でも読み込めるようにする
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # モデル全体が保存されているか、state_dictだけかを確認してロード
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and keys_match(model, checkpoint):
             model.load_state_dict(checkpoint)
        else:
             # そのままロードを試みる
             model.load_state_dict(checkpoint)
             
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # -------------------------------------------------
    # 4. テストデータの生成
    # -------------------------------------------------
    print("Generating test data...")
    # 可視化に必要な長さ: init + seq + 予備
    SEQ_LEN_GEN = config.init_size + config.seq_size + 5
    
    # 毎回異なるデータを生成して確認したい場合はここを呼び出す
    test_data = generate_vta_dataset(max(args.num_samples, args.idx + 1), seq_len=SEQ_LEN_GEN, size=32)
    
    # utils.pyのvisualize_resultsは batch = next(iter(loader)) で最初のバッチしか見ないため
    # バッチサイズを大きくして、指定された idx が含まれるようにする
    test_loader = DataLoader(test_data, batch_size=args.num_samples, shuffle=False)

    # -------------------------------------------------
    # 5. 可視化の実行
    # -------------------------------------------------
    print(f"Visualizing sequence index: {args.idx}")
    
    # 保存先ファイル名を変更したい場合、config.work_dir を一時的に変更するか、
    # visualize_results 実行後にファイルを移動するなどの工夫が可能ですが、
    # ここでは utils.py の仕様通り config.work_dir に保存させます。
    
    visualize_results(model, test_loader, config, seq_idx=args.idx)
    
    print(f"Done. Check the output image in: {config.work_dir}/vis_seq_{args.idx}.png")

def keys_match(model, state_dict):
    """state_dictのキーがモデルと一致するか簡易チェック"""
    model_keys = set(model.state_dict().keys())
    dict_keys = set(state_dict.keys())
    return len(model_keys.intersection(dict_keys)) > 0

if __name__ == "__main__":
    main()