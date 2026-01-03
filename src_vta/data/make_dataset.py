'''
3D_Maze環境のデータセットを読み込み，PyTorchのDataLoaderで扱える形式に変換するモジュール
(現在trainm.pyの実装にも一部関数を取り入れているため，こちらは使用していない．今後こちらを使用するように修正予定)
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.utils.tensorboard import SummaryWriter
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from pathlib import Path
import glob
from tqdm.notebook import tqdm
from typing import Optional, Union

from src_vta.data.paths import resolve_maze_data_dir

# 離散行動の数（左折，右折，直進で3）
ACTION_SIZE = 3

class MazeDataset(torch.utils.data.Dataset):
    """
    Colabでは__getitem__内でドライブからデータをロードすると学習が大きく遅延する場合があるため，
    メモリを圧迫しますがすべてのサンプルをはじめにすべてロードします．
    """
    def __init__(
        self,
        length,
        partition="train",
        image_width=32,
        image_height=32,
        image_channels=3,
        one_hot_action=True,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Parameters
        ----------
            length: モデル入力として切り出す動画のタイムステップ数
            partition: データの訓練・テストスプリット
            image_width, image_height, image_channels: モデル入力の画像サイズ
            one_hot_action: アクションをone-hot表現に変換するかどうか
        """
        self.data_dir = resolve_maze_data_dir(data_dir)
        self.partition = partition
        self.length = length
        self.height = image_height
        self.width = image_width
        self.image_channels = image_channels
        self.one_hot_action = one_hot_action

        self._load_data()

    def _load_data(self) -> None:
        """ すべてのnpzファイルから画像系列と行動系列をメモリに読み込む """
        # Find .npz files
        dir_path = Path(self.data_dir) / self.partition
        self.file_paths = sorted(glob.glob(str(dir_path / "*.npz")))
        assert len(self.file_paths) > 0, f"Dataset not found: {dir_path}"

        self.data = {"video": [], "actions": []}
        for path in tqdm(self.file_paths, desc=f"Loading {self.partition} samples."):
            sample_episode = np.load(path, allow_pickle=True)
            frames = self._resize(sample_episode["video"])
            self.data["video"].append(frames) # ( t, h, w, c )
            self.data["actions"].append(sample_episode["actions"]) # ( t, )

        self.data["video"] = np.stack(self.data["video"], axis=0)
        self.data["actions"] = np.stack(self.data["actions"], axis=0)

        print(f"Dataset ({self.partition}): {self.data['video'].shape[0]} episodes, \
            videos: {self.data['video'].shape}, actions: {self.data['actions'].shape}")

    def _resize(self, frames: np.ndarray):
        """
        画像系列をリサイズ

        Parameters
        ----------
            frames ( t, h, w, c ): リサイズ前（元データ）の画像系列
        Returns
        ----------
            frames ( t, h_resized, w_resized, c ): リサイズ後（モデル入力）の画像系列
        """
        return np.stack(
            [cv2.resize(frame, (self.width, self.height)) for frame in frames]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        """ エピソードからself.lengthの系列を切り出し，前処理する """
        frames = torch.from_numpy(self.data["video"][index])
        actions = torch.from_numpy(self.data["actions"][index])

        seq_len = frames.shape[0]

        if seq_len > self.length:
            start_t = np.random.randint(0, seq_len - self.length)
        else:
            start_t = 0

        frames = frames[start_t:start_t + self.length]
        actions = actions[start_t:start_t + self.length]

        # (T, H, W, C) -> (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
        # [0, 255] -> [0, 1]
        frames = frames.float() / 255.0

        # 行動をone-hot表現にする
        if self.one_hot_action:
            actions = torch.nn.functional.one_hot(
                actions.long(), num_classes=ACTION_SIZE
            )

        return frames, actions


def full_dataloader(seq_size, init_size, batch_size, test_size=16):
    length = seq_size + init_size * 2
    train_loader = MazeDataset(length, partition="train")
    test_loader = MazeDataset(length, partition="test")

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader, test_size, shuffle=False)

    return train_loader, test_loader

# display_video関数を以下のように書き換えてください

def display_video(video: torch.Tensor, figsize: tuple = (8, 8), save_path="output.gif") -> None:
    """
    動画を可視化して保存する関数
    """
    # テンソルがGPUにある場合はCPUに戻し、numpyに変換
    if isinstance(video, torch.Tensor):
        video = video.cpu().detach().numpy()
        
    # 値が0-1の範囲なら0-255に戻す（もし必要なら）
    if video.max() <= 1.0:
        video = (video * 255).astype(np.uint8)
    else:
        video = video.astype(np.uint8)

    fig = plt.figure(figsize=figsize, dpi=40, tight_layout=True)
    patch = plt.imshow(video[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(video[i])
        plt.title("Step %d" % (i))

    anim = animation.FuncAnimation(fig, animate, frames=len(video), interval=50)
    
    # Jupyter上の表示用（インポートしていれば機能します）
    # from IPython.display import HTML, display
    # display(HTML(anim.to_jshtml(default_mode='once')))
    
    # ★ファイルとして保存（こちらを推奨）
    anim.save(save_path, writer='pillow', fps=20)
    print(f"Animation saved to {save_path}")
    
    plt.close()
    
vis_len = 300 # 最大300

test_loader = torch.utils.data.DataLoader(
    MazeDataset(vis_len, partition="test"), batch_size=1, shuffle=True
)

video, _ = next(iter(test_loader)) # ( b, t, c, h, w )
video = video[0].permute(0, 2, 3, 1) # ( t, h, w, c )

display_video(video)
