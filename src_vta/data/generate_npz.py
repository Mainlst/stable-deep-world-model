'''
3D_Maze環境のデータセットを生成し、.npzファイルとして保存するスクリプト
(現在3D_Maze環境はうまく設計できていないため，使用予定なし)
'''

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 作成済みの環境をインポート
from .maze_env import MazeEnv
from .paths import DEFAULT_MAZE_DIR, resolve_maze_data_dir

def generate_and_save(
    save_dir: Path,
    num_episodes: int,
    seq_len: int = 50,
    resolution: int = 32
):
    """
    指定された数のエピソードを生成し、個別の.npzファイルとして保存する
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    env = MazeEnv(resolution=resolution)
    
    print(f"Generating {num_episodes} episodes into {save_dir}...")
    
    for i in tqdm(range(num_episodes)):
        env.reset()
        
        frames = []
        actions = []
        
        # 1エピソード分のループ
        for _ in range(seq_len):
            obs = env.get_observation()
            obs_uint8 = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
            frames.append(obs_uint8)
            
            # --- 厳密な行動決定ロジック ---
            # 環境から「現在許可されているアクション」のリストを取得
            valid_actions = env.get_permissible_actions()
            
            if len(valid_actions) == 0:
                # 万が一動けない場合（ありえないが安全策）、回転させる
                action = np.random.choice([1, 2])
            elif 0 in valid_actions and len(valid_actions) > 1:
                # 直進も回転もできる場合（交差点）
                # 論文: "randomly navigate" -> ランダムに選択
                # ただし、直進の確率を少し上げるとスムーズに見える（任意）
                action = np.random.choice(valid_actions)
            elif 0 in valid_actions and len(valid_actions) == 1:
                # 直進しかない場合（廊下）-> 迷わず直進
                action = 0
            else:
                # 回転しかない場合（突き当たり）-> ランダムに回転
                action = np.random.choice(valid_actions)
            
            env.step(action)
            actions.append(action)
            
        # numpy配列に変換
        frames_np = np.array(frames)   # (T, H, W, C)
        actions_np = np.array(actions) # (T,)
        
        # .npzファイルとして保存
        # MazeDataset._load_data は "video" と "actions" キーを期待している
        file_path = save_dir / f"episode_{i:05d}.npz"
        np.savez_compressed(
            file_path,
            video=frames_np,
            actions=actions_np
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Generate 3D Maze npz datasets.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MAZE_DIR,
        help="Base directory to place the generated dataset (train/test).",
    )
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of test episodes.")
    parser.add_argument("--seq_len", type=int, default=300, help="Frames per episode.")
    parser.add_argument("--resolution", type=int, default=32, help="Render resolution.")
    return parser


def main():
    args = build_parser().parse_args()
    ROOT_DIR = resolve_maze_data_dir(args.out)
    TRAIN_EPISODES = args.train_episodes  # 学習用エピソード数（必要に応じて増減してください）
    TEST_EPISODES = args.test_episodes    # テスト用エピソード数
    SEQ_LEN = args.seq_len                # 1エピソードあたりのフレーム数
    RESOLUTION = args.resolution          # 画像サイズ
    
    # Trainデータの生成
    generate_and_save(
        save_dir=ROOT_DIR / "train",
        num_episodes=TRAIN_EPISODES,
        seq_len=SEQ_LEN,
        resolution=RESOLUTION
    )
    
    # Testデータの生成
    generate_and_save(
        save_dir=ROOT_DIR / "test",
        num_episodes=TEST_EPISODES,
        seq_len=SEQ_LEN,
        resolution=RESOLUTION
    )
    
    print("All datasets generated successfully.")
    print(f"Train data: {ROOT_DIR}/train/*.npz")
    print(f"Test data:  {ROOT_DIR}/test/*.npz")

if __name__ == "__main__":
    main()
