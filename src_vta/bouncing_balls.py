import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset

class VTABouncingBalls:
    """
    VTA論文に基づくBouncing Balls環境
    ボールの形状を円ではなく「ひし形」で描画するバージョン
    """
    def __init__(self, size=32, num_balls=2, radius=3, dt=1.0, noise_scale=0.1):
        self.size = size
        self.num_balls = num_balls
        self.radius = radius
        self.dt = dt
        self.noise_scale = noise_scale
        
        self.colors_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        self.reset()

    def reset(self):
        # 位置をランダムに初期化（壁にめり込まない範囲）
        self.pos = np.random.uniform(self.radius, self.size - self.radius, (self.num_balls, 2))
        self.vel = np.random.randn(self.num_balls, 2)
        norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = (self.vel / (norm + 1e-8)) * 1.5 
        
        self.current_colors = []
        for _ in range(self.num_balls):
            color = self.colors_palette[np.random.randint(len(self.colors_palette))]
            self.current_colors.append(color)
            
        return self.render()

    def step(self):
        self.pos += self.vel * self.dt
        self.vel += np.random.randn(*self.vel.shape) * self.noise_scale
        hit_any = False # このフレームで衝突があったかフラグ
        
        for i in range(self.num_balls):
            hit = False
            # X軸の反射判定
            if self.pos[i, 0] <= self.radius:
                self.pos[i, 0] = self.radius
                self.vel[i, 0] *= -1
                hit = True
            elif self.pos[i, 0] >= self.size - self.radius:
                self.pos[i, 0] = self.size - self.radius
                self.vel[i, 0] *= -1
                hit = True
                
            # Y軸の反射判定
            if self.pos[i, 1] <= self.radius:
                self.pos[i, 1] = self.radius
                self.vel[i, 1] *= -1
                hit = True
            elif self.pos[i, 1] >= self.size - self.radius:
                self.pos[i, 1] = self.size - self.radius
                self.vel[i, 1] *= -1
                hit = True
            
            if hit:
                self._change_color(i)
                hit_any = True

        return self.render(), hit_any # 画像と衝突フラグを返す

    def _change_color(self, ball_idx):
        # 1. 現在の色の取得
        current = self.current_colors[ball_idx]
        # 2. 次の色の候補リストを作成（現在の色を除外）
        candidates = [c for c in self.colors_palette if c != current]
        # 3. 候補の中からランダムに1つ選んで更新
        self.current_colors[ball_idx] = candidates[np.random.randint(len(candidates))]

    def render(self):
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for i in range(self.num_balls):
            x, y = self.pos[i].astype(int)
            r = self.radius
            c = self.current_colors[i]

            # ひし形の頂点を計算 (上, 右, 下, 左)
            # OpenCVのfillPolyで描画するために頂点配列を作成
            pts = np.array([
                [x, y - r],  # Top
                [x + r, y],  # Right
                [x, y + r],  # Bottom
                [x - r, y]   # Left
            ], np.int32)
            
            pts = pts.reshape((-1, 1, 2))
            
            # ポリゴン（ひし形）の塗りつぶし描画
            cv2.fillPoly(img, [pts], c)
        return img

def generate_vta_dataset(num_sequences, seq_len=30, size=32, dt=2.0):
    print(f"Generating {num_sequences} sequences (len={seq_len})...")
    env = VTABouncingBalls(size=size, dt=dt)
    
    dataset_imgs = []
    dataset_hits = [] # ★追加: 衝突フラグを保存するリスト
    
    for _ in range(num_sequences):
        frames = []
        hits = [] # ★追加
        
        img = env.reset()
        frames.append(img)
        hits.append(False) # 初期フレームは衝突なしとする

        for _ in range(seq_len - 1):
            img, hit = env.step() # ★変更: 戻り値を受け取る
            frames.append(img)
            hits.append(hit)
        
        # (Seq, H, W, C) -> (Seq, C, H, W)
        seq_data = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        dataset_imgs.append(seq_data)
        dataset_hits.append(np.array(hits, dtype=np.float32)) # ★追加
    
    # データをテンソル化
    imgs_tensor = torch.from_numpy(np.array(dataset_imgs, dtype=np.float32) / 255.0)
    hits_tensor = torch.from_numpy(np.array(dataset_hits, dtype=np.float32)) # (B, T)
    
    # Datasetとして扱いやすいように TensorDataset にまとめる
    return TensorDataset(imgs_tensor, hits_tensor)

def get_dataloaders(config, dt=2.0):
    # init_size + seq_size 分が必要なので余裕を持って生成
    gen_len = config.init_size + config.seq_size + 5
    
    train_data = generate_vta_dataset(2000, seq_len=gen_len, dt=dt)
    test_data = generate_vta_dataset(500, seq_len=gen_len, dt=dt)
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader