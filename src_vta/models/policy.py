import torch
import numpy as np

class VTAPolicy:
    """
    VTAエージェントを用いて推論（テスト実行）を行うためのラッパークラス。
    環境からの観測を1ステップずつ受け取り、内部状態(Belief/State)を維持しながら行動を決定する。
    """
    def __init__(self, agent, device):
        self.agent = agent
        self.rssm = agent.vta.state_model
        self.actor = agent.actor
        self.device = device
        
        # 評価モードに設定（学習ループ内で一時的に使う場合は注意が必要だが、
        # 推論専用オブジェクトとして使うならここでevalにしておくのが安全）
        self.rssm.eval()
        self.actor.eval()
        
        self.reset()

    def reset(self):
        """内部状態と前回の行動をリセットする"""
        # バッチサイズは1（シングル環境）を想定
        batch_size = 1
        
        # 抽象状態 (Abstract Level)
        self.abs_belief = torch.zeros(batch_size, self.rssm.abs_belief_size).to(self.device)
        self.abs_state = torch.zeros(batch_size, self.rssm.abs_state_size).to(self.device)
        
        # 前回の行動 (初期値はゼロ)
        self.prev_action = torch.zeros(batch_size, self.rssm.act_size).to(self.device)
        
        # 最初のステップであることを示すフラグ (最初は必ず更新するため)
        self.is_first_step = True
        
        # RNNのHidden State等の初期化が必要であればここで行う
        # 例: self.rssm.init_abs_belief(...) など

    def __call__(self, obs, eval_mode=False):
        """
        環境からの観測を受け取り、行動を返す
        Args:
            obs: (C, H, W) の形状を持つ観測データ (Numpy array or Tensor)
            eval_mode: Trueなら決定的(mean)、Falseなら確率的(sample)に行動を選択
        Returns:
            action: (ActDim,) のNumpy配列
        """
        with torch.no_grad():
            # 1. 観測の前処理 (Tensor化 & バッチ次元追加)
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float().to(self.device)
            if obs.ndim == 3:
                obs = obs.unsqueeze(0) # (1, C, H, W)
                
            import pdb; pdb.set_trace()
                
            # 2. 観測のエンコード
            # Encoderは通常 (B, C, H, W) を受け取る
            # PostBoundaryDetectorへの入力用に時間次元を追加: (B, T, Feat) -> (1, 1, Feat)
            enc_obs = self.rssm.enc_obs(obs).unsqueeze(1)

            # 3. 境界検知 (m_t)
            post_boundary_logits = self.rssm.post_boundary(enc_obs)
            boundary_sample, _ = self.rssm.boundary_sampler(post_boundary_logits)
            
            # boundary_sample: (B, T, 2) -> read=0, copy=1
            read_flag = boundary_sample[0, 0, 0].item() # 1.0 or 0.0

            # 4. 状態更新と行動決定
            if read_flag > 0.5 or self.is_first_step:
                # --- Update Step ---
                
                # 抽象Beliefの更新 (入力: 前回の状態 + 前回の行動)
                abs_input = torch.cat([self.abs_state, self.prev_action], dim=1)
                self.abs_belief = self.rssm.update_abs_belief(abs_input, self.abs_belief)
                
                # 抽象Stateのサンプリング (オンライン推論時はPriorを使用)
                prior_dist = self.rssm.prior_abs_state(self.abs_belief)
                self.abs_state = prior_dist.sample()
                
                # 特徴量の作成
                abs_feat = self.rssm.abs_feat(torch.cat([self.abs_belief, self.abs_state], dim=1))
                
                # Actorによる行動選択
                action_dist = self.actor(abs_feat)
                if eval_mode:
                    action = action_dist.mean # 評価時は平均値（決定的）
                else:
                    action = action_dist.sample() # 探索時はサンプリング
                
                self.is_first_step = False
                
            else:
                # --- Copy Step (Action Repeat) ---
                # 状態更新をスキップし、前回の行動を継続
                action = self.prev_action

            # 5. 次のステップのために行動を保存
            self.prev_action = action
            
            return action.cpu().numpy()[0]