import os
import gymnasium as gym
import numpy as np
import imageio
from datetime import datetime

class EpisodeRecoderWrapper(gym.Wrapper):
    """
    エピソードごとに観測フレームを保存し，GIFファイルとして出力するGymラッパー．
    """
    def __init__(self, env, save_dir, rank=0, save_interval=50, max_frames=1000):
        """
        Args:
            env: Gym 環境
            save_dir: 録画データの保存ディレクトリ
            rank: 環境のランク (並列環境の場合)
            save_interval: 何エピソードごとに録画を保存するか
            max_frames: 録画する最大フレーム数
        """
        super().__init__(env)
        self.save_dir = save_dir
        self.rank = rank
        self.save_interval = save_interval
        self.frames = []
        self.episode_count = 0
        self.max_frames = max_frames
        
        if self.rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # rank == 0の場合のみフレームを保存します．
        if self.rank == 0 and (self.episode_count % self.save_interval == 0):
            frame = obs.copy()
            
            # (H, W, C), uint8に変換します.
            if frame.shape[0] in [1, 3]:
                frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
            if frame.dtype == np.float32 and frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
                
            self.frames.append(frame)
        
        # エピソード終了時/中断時にGIFを保存します．
        if terminated or truncated:
            if self.rank == 0 and (self.episode_count % self.save_interval == 0):
                self._save_gif()
            self.frames = []
            self.episode_count += 1
            
        return obs, reward, terminated, truncated, info
    
    def _save_gif(self):
        if len(self.frames) == 0:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"ep_{self.episode_count}_rank_{self.rank}_{timestamp}.gif")
        
        try:
            imageio.mimsave(filename, self.frames[:self.max_frames], fps=30)
            print(f"Saved GIF: {filename}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")