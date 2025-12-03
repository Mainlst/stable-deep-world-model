# VTA論文の実験設定をまとめた設定ファイル

import torch
from pathlib import Path

class Config:
    def __init__(self):
        self.exp_name = "vta_bouncing_balls"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- ディレクトリ設定 ---
        self.work_dir = Path("./experiments5_test")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # シード設定
        self.seed = 111
        
        # --- ★データセット生成モード設定★ ---
        # "fixed": 最初に大量に生成し、それを使い回す (過学習の確認用 / メモリ大)
        # "infinite": エポックごとに少量生成し、常に新しいデータを使う (過学習回避 / メモリ小)
        self.data_mode = "fixed"
        
        # 1回の生成で作るデータ数
        # infiniteモードなら 2000 程度 (メモリ節約)
        # fixedモードなら 20000~50000 程度 (過学習しないよう多めに確保が必要)
        self.epoch_data_size = 2_000 if self.data_mode == "infinite" else 45_000

        # --- データサイズ関連 ---
        self.batch_size = 64
        self.seq_size = 20        # 学習対象の系列長
        self.init_size = 5        # 初期状態を決めるためのコンテキスト長
        self.action_size = 0      # Bouncing Ballsは行動なし

        # --- モデルサイズ関連 ---
        self.state_size = 8       # 潜在状態zおよびsの次元数
        self.belief_size = 128    # 信念状態（RNNの隠れ状態）の次元数
        self.num_layers = 3       # PostBoundaryDetectorの中間層の数

        # --- ★実験設定のプリセット切り替え★ ---
        # "bouncing_balls" または "3d_maze" (想定)
        self.env_type = "bouncing_balls"
        self.env_type = "3d_maze"

        if self.env_type == "bouncing_balls":
            self.loss_type = "bce"    # くっきり生成
            self.obs_std = 1.0        # BCEでは無視される
            self.obs_bit = None       # 前処理なし (0.0-1.0のまま)
            self.dt = 2.0             # 環境の時間刻み幅
        else:
            # 3D Maze等の場合
            self.loss_type = "mse"    # ガウス分布
            self.obs_std = 1.0        # 標準偏差
            self.obs_bit = 5          # ビット深度を5bitに削減
            self.dt = 1.0             # ダミー
            self.action_size = 3      # アクションサイズを3に設定

        # --- 最適化関連 ---
        self.learn_rate = 5e-4     # 学習率
        self.grad_clip = 10.0      # 勾配クリッピングの閾値
        self.max_iters = 100_000    # 最大学習イテレーション数

        # --- 部分系列の事前分布に関する制約 ---
        self.seg_num = 5          # 1系列あたりの部分系列の最大数
        self.seg_len = 8         # 部分系列の最大長

        # --- Gumbel-Softmax関連 ---
        self.max_beta = 1.0       # Gumbel-Softmaxの温度パラメータの最大値
        self.min_beta = 0.1       # Gumbel-Softmaxの温度パラメータの最小値
        self.beta_anneal = 100    # 温度を最大値から最小値にアニーリングする際の減衰率