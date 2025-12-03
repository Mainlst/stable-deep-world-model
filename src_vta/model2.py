'''
VTAの主要なモデルコンポーネントを実装したファイル(修正前)
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence

class LatentDistribution(nn.Module):
    """ 潜在変数zまたはsの分布（平均と分散）をパラメータ化するネットワーク """
    def __init__(self, input_size, latent_size, feat_size=None):
        """
        Parameters
        ----------
            input_size (int): 入力特徴量の次元数
            latent_size (int): 潜在変数の次元数
            feat_size (int, optional): 中間層の特徴量の次元数．Noneの場合中間層なし
        """
        super(LatentDistribution, self).__init__()

        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = nn.Sequential(
                nn.Linear(input_size, feat_size), nn.ELU(inplace=True)
            )

        # 平均を出力する線形層
        self.mean = nn.Linear(feat_size, latent_size)

        # 標準偏差を出力する線形層（Sigmoidで0〜1の範囲に正規化）
        self.std = nn.Sequential(
            nn.Linear(feat_size, latent_size), nn.Sigmoid()
        )

    def forward(self, input_data):
        """
        Parameters
        ----------
            input_data (torch.Tensor): ( b, input_size )
        Returns
        ----------
            torch.distributions.Normal: パラメータ化された正規分布
        """
        feat = self.feat(input_data)
        return Normal(loc=self.mean(feat), scale=self.std(feat))


class Encoder(nn.Module):
    """ 観測画像を低次元の特徴量ベクトルにエンコードするCNN """
    def __init__(self, output_size=None, feat_size=64):
        """
        Parameters
        ----------
            output_size (int, optional): 最終出力の次元数．Noneの場合feat_sizeと同じ
            feat_size (int): 畳み込み層の基本特徴量マップ数
        """
        super(Encoder, self).__init__()

        network_list = []
        num_layers = 4
        # 4層の畳み込みネットワークを構築
        for l in range(num_layers):
            input_size = 3 if l == 0 else feat_size
            is_final_layer = l == num_layers - 1
            network_list.append(
                nn.Conv2d(
                    input_size,
                    feat_size,
                    kernel_size=4,
                    stride=1 if is_final_layer else 2,  # 最後だけストライド1
                    padding=0 if is_final_layer else 1,
                )
            )
            network_list.append(nn.BatchNorm2d(feat_size))
            network_list.append(nn.ELU(inplace=True))
        # 1次元ベクトルに平坦化
        network_list.append(nn.Flatten())

        # 指定されていれば，最後に追加の全結合層を適用
        if output_size is not None:
            network_list.append(nn.Linear(feat_size, output_size))
            network_list.append(nn.ELU(inplace=True))
            self.output_size = output_size
        else:
            self.output_size = feat_size

        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        """ 画像をエンコード

        Parameters
        ----------
            input_data (torch.Tensor): ( b, c, h, w )
        Returns
        ----------
            torch.Tensor: ( b, output_size )
        """
        return self.network(input_data)


class Decoder(nn.Module):
    """ 特徴量ベクトルから観測画像をデコード（再構成）する逆畳み込みCNN """
    def __init__(self, input_size, feat_size=64):
        """
        Parameters
        ----------
            input_size (int): 入力特徴量ベクトルの次元数
            feat_size (int): 逆畳み込み層の基本特徴量マップ数
        """
        super(Decoder, self).__init__()

        # 入力次元と特徴量次元が異なる場合，線形層で変換
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(input_size, feat_size)

        network_list = []
        num_layers = 4
        # 4層の逆畳み込みネットワークを構築
        for l in range(num_layers):
            is_final_layer = l == num_layers - 1

            network_list.append(
                nn.ConvTranspose2d(
                    feat_size,
                    3 if is_final_layer else feat_size,  # 最後の層の出力チャネルは3 (RGB)
                    kernel_size=4,
                    stride=1 if l == 0 else 2,  # 最初だけストライド1
                    padding=0 if l == 0 else 1,
                )
            )

            # 最終層の活性化関数はTanh，それ以外はBatchNorm + ELU
            if is_final_layer:
                network_list.append(nn.Tanh())
            else:
                network_list.append(nn.BatchNorm2d(feat_size))
                network_list.append(nn.ELU(inplace=True))

        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        """ 画像を再構成

        Parameters
        ----------
            input_data (torch.Tensor): ( b, input_size )
        Returns
        ----------
            torch.Tensor: ( b, c, h, w )
        """
        # unsqueezeで空間次元を追加してからネットワークに入力
        return self.network(self.linear(input_data).unsqueeze(-1).unsqueeze(-1))


class PriorBoundaryDetector(nn.Module):
    """ 事前分布 p(m_t|s_t) をモデル化するネットワーク．観測状態 s_t から境界のロジットを出力する """
    def __init__(self, input_size, output_size=2):
        """
        Parameters
        ----------
            input_size (int): 入力（観測状態s_tに由来する特徴量）の次元数
            output_size (int): 出力（境界のロジット）の次元数．通常は2
        """
        super(PriorBoundaryDetector, self).__init__()
        self.network = nn.Linear(input_size, output_size)

    def forward(self, input_data):
        """
        Parameters
        ----------
            input_data (torch.Tensor): ( b, input_size )
        Returns
        ----------
            torch.Tensor: 境界のロジット．( b, 2 )
        """
        return self.network(input_data)


class PostBoundaryDetector(nn.Module):
    """ 事後分布 q(m|x) をモデル化するネットワーク．観測系列 x 全体から各ステップの境界ロジットを出力する """
    def __init__(self, input_size, output_size=2, num_layers=1):
        """
        Parameters
        ----------
            input_size (int): 入力（エンコードされた観測）の次元数
            output_size (int): 出力（境界のロジット）の次元数．通常は2
            num_layers (int): 中間層の数
        """
        super(PostBoundaryDetector, self).__init__()

        network = list()
        # 時間軸に沿った1D畳み込み層を重ねる
        for l in range(num_layers):
            network.append(
                nn.Conv1d(input_size, input_size, 3, stride=1, padding=1, bias=False)
            )
            network.append(nn.BatchNorm1d(input_size))
            network.append(nn.ELU(inplace=True))
        # 最終層で出力次元を2に変換
        network.append(nn.Conv1d(input_size, output_size, 3, stride=1, padding=1))
        self.network = nn.Sequential(*network)

    def forward(self, input_data_list):
        """
        Parameters
        ----------
            input_data_list (torch.Tensor): エンコードされた観測系列．( b, t, d )
        Returns
        ----------
            torch.Tensor: 各ステップの境界のロジット．( b, t, 2 )
        """
        input_data = input_data_list.permute(0, 2, 1) # ( b, d, t )
        return self.network(input_data).permute(0, 2, 1) # ( b, t, 2 )
    
class HierarchicalRSSM(nn.Module):
    def __init__(
        self,
        belief_size,
        state_size,
        act_size,
        num_layers,
        max_seg_len,
        max_seg_num,
    ):
        super(HierarchicalRSSM, self).__init__()
        # --- ネットワークのサイズ定義 ---
        # 抽象レベル
        self.abs_belief_size = belief_size
        self.abs_state_size = state_size
        self.abs_feat_size = belief_size

        # 観測レベル
        self.obs_belief_size = belief_size
        self.obs_state_size = state_size
        self.obs_feat_size = belief_size

        # その他のサイズ
        self.num_layers = num_layers
        self.feat_size = belief_size
        self.act_size = act_size

        # 部分系列の制約情報
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        # Gumbel-Softmax (Concrete分布) で使用する温度パラメータ
        self.mask_beta = 1.0

        # --- 観測エンコーダ・デコーダ ---
        self.enc_obs = Encoder(feat_size=self.feat_size)
        self.dec_obs = Decoder(
            input_size=self.obs_feat_size, feat_size=self.feat_size
        )

        # --- 境界検出器 ---
        self.prior_boundary = PriorBoundaryDetector(self.obs_feat_size)
        self.post_boundary = PostBoundaryDetector(
            self.feat_size, num_layers=self.num_layers
        )

        # --- 特徴量抽出器 ---
        self.abs_feat = nn.Linear(
            self.abs_belief_size + self.abs_state_size, self.abs_feat_size
        )
        self.obs_feat = nn.Linear(
            self.obs_belief_size + self.obs_state_size, self.obs_feat_size
        )

        # --- 信念状態の初期化 ---
        self.init_abs_belief = nn.Identity()
        self.init_obs_belief = nn.Identity()

        # --- 信念状態の更新 (RNN) ---
        self.update_abs_belief = nn.GRUCell(
            self.abs_state_size + self.act_size, self.abs_belief_size
        )
        self.update_obs_belief = nn.GRUCell(
            self.obs_state_size + self.abs_feat_size, self.obs_belief_size
        )

        # --- 事後分布エンコーダ (双方向RNN) ---
        self.abs_post_fwd = nn.GRUCell(self.feat_size, self.abs_belief_size)
        self.abs_post_bwd = nn.GRUCell(self.feat_size, self.abs_belief_size)
        self.obs_post_fwd = nn.GRUCell(self.feat_size, self.obs_belief_size)

        # --- 状態の事前分布 ---
        self.prior_abs_state = LatentDistribution(
            input_size=self.abs_belief_size, latent_size=self.abs_state_size
        )
        self.prior_obs_state = LatentDistribution(
            input_size=self.obs_belief_size, latent_size=self.obs_state_size
        )

        # --- 状態の事後分布 ---
        self.post_abs_state = LatentDistribution(
            input_size=self.abs_belief_size + self.abs_belief_size,
            latent_size=self.abs_state_size,
        )
        self.post_obs_state = LatentDistribution(
            input_size=self.obs_belief_size + self.abs_feat_size,
            latent_size=self.obs_state_size,
        )

    @staticmethod
    def gumbel_sampling(log_alpha, temp, margin=1e-4):
        """ Gumbel分布からのノイズを利用し，カテゴリカル分布からのサンプリングを模倣

        Parameters
        ----------
            log_alpha (torch.Tensor): サンプリング対象のロジット（正規化されていない対数確率）
            temp (float): 温度パラメータ．小さいほどone-hotに近い
            margin (float): 数値的安定性のための微小値
        Returns
        ----------
            torch.Tensor: Gumbelノイズが加算され，温度でスケールされたロジット
        """
        # Gumbel(0, 1)からのノイズを生成
        noise = log_alpha.new_empty(log_alpha.shape).uniform_(margin, 1 - margin)
        gumbel_sample = -torch.log(-torch.log(noise))

        return torch.div(log_alpha + gumbel_sample, temp)

    def boundary_sampler(self, log_alpha):
        """ 部分系列の境界（終了するか継続するか）を微分可能な形でサンプリングする

        Parameters
        ----------
            log_alpha (torch.Tensor): 部分系列が終了(READ)するか継続(COPY)するかのロジット．(b, t, 2)
        Returns
        ----------
            sample_data (torch.Tensor): サンプリングされたone-hot表現．(b, t, 2)
            log_sample_alpha (torch.Tensor): サンプリングに用いられた対数確率．(b, t, 2)
        """
        # Gumbel-Softmax Trickによる微分可能なサンプリング
        if self.training:
            log_sample_alpha = self.gumbel_sampling(
                log_alpha=log_alpha, temp=self.mask_beta
            )
        else:
            # 評価時は決定的に最大値を選択
            log_sample_alpha = log_alpha / self.mask_beta

        # Softmaxを計算し確率に変換（数値的に安定なlogsumexpトリックを使用）
        log_sample_alpha = log_sample_alpha - torch.logsumexp(
            log_sample_alpha, dim=-1, keepdim=True
        )
        # 指数関数を適用し，実際の確率分布を取得
        sample_prob = log_sample_alpha.exp()  # ( b, t, 2 )

        # 確率が最大となるインデックスを取得し，one-hotベクトルに変換
        sample_data = torch.eye(
            2, dtype=log_alpha.dtype, device=log_alpha.device
        )[torch.max(sample_prob, dim=-1)[1]]  # ( b, t, 2 )

        # 順伝播では離散的なサンプル(sample_data)を使い，
        # 逆伝播では連続的な確率(sample_prob)の勾配を流す (Straight-Through Estimator)
        sample_data = sample_data.detach() + (sample_prob - sample_prob.detach()) # WRITE ME

        return sample_data, log_sample_alpha

    def regularize_prior_boundary(self, log_alpha_list, boundary_data_list):
        """境界の事前分布に対して，部分系列の最大長や最大数といった制約を適用する．

        Parameters
        ----------
            log_alpha_list (torch.Tensor): 事前分布のロジット．( b, t, 2 )
            boundary_data_list (torch.Tensor): サンプリングされた境界のone-hot表現．( b, t, 2 )
        Returns
        ----------
            torch.Tensor: 制約が適用された後のロジット．( b, t, 2 )
        """
        # 学習時のみ適用
        if not self.training:
            return log_alpha_list

        num_samples = boundary_data_list.shape[0]
        seq_len = boundary_data_list.shape[1]

        # セグメントの状態（現在のセグメント数と長さ）を初期化
        seg_num = log_alpha_list.new_zeros(num_samples, 1)
        seg_len = log_alpha_list.new_zeros(num_samples, 1)

        # 制約を強制するための，ほぼ1またはほぼ0の確率に対応するロジットを準備
        one_prob = 1 - 1e-3
        max_scale = np.log(one_prob / (1 - one_prob))

        # READを強制するためのロジット ([1, 0]に対応)
        near_read_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_read_data[:, 1] = - near_read_data[:, 1]
        # COPYを強制するためのロジット ([0, 1]に対応)
        near_copy_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_copy_data[:, 0] = - near_copy_data[:, 0]

        # 各時間ステップで処理
        new_log_alpha_list = []
        for t in range(seq_len):
            # (0) 現在のセグメント長とセグメント数を更新
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)
            seg_len = read_data * 1.0 + copy_data * (seg_len + 1.0)
            seg_num = read_data * (seg_num + 1.0) + copy_data * seg_num
            over_len = torch.ge(seg_len, self.max_seg_len).float().detach()
            over_num = torch.ge(seg_num, self.max_seg_num).float().detach()

            # (1) 制約に基づいてlog_alphaを正則化
            # セグメント数が上限に達したら，COPYを強制
            new_log_alpha = over_num * near_copy_data + (1.0 - over_num) * log_alpha_list[:, t]

            # セグメント長が上限に達したら，READを強制
            new_log_alpha = over_len * near_read_data + (1.0 - over_len) * new_log_alpha

            # (2) 更新されたロジットを保存
            new_log_alpha_list.append(new_log_alpha)

        return torch.stack(new_log_alpha_list, dim=1)

    @staticmethod
    def log_density_concrete(log_alpha, log_sample, temp):
        """Concrete (Gumbel-Softmax)分布の対数確率密度を計算

        Parameters
        ----------
            log_alpha (torch.Tensor): 分布のパラメータであるロジット
            log_sample (torch.Tensor): 評価点（サンプルの対数確率）
            temp (float): 温度パラメータ
        Returns
        ----------
            torch.Tensor: 計算された対数確率密度
        """
        exp_term = log_alpha - temp * log_sample
        log_prob = torch.sum(exp_term, -1) - 2.0 * torch.logsumexp(exp_term, -1)
        return log_prob

    def forward(
        self,
        obs_data_list: torch.Tensor,
        act_data_list: torch.Tensor,
        seq_size: int,
        init_size: int,
    ):
        """観測系列と行動系列からELBOを計算するための順伝播

        Parameters
        ----------
            obs_data_list (torch.Tensor): 観測系列．( b, t, c, h, w )
            act_data_list (torch.Tensor): 行動系列．( b, t, action_size )
            seq_size (int): 学習対象の系列長
            init_size (int): モデルの初期状態を決定するためのコンテキスト長
        Returns
        ----------
            list: 再構成された観測やKLダイバージェンスなど，損失計算に必要なテンソルのリスト
        """
        num_samples, full_seq_size = obs_data_list.shape[:2]

        # 観測をエンコード (時間次元とバッチ次元を一時的に統合して効率化)
        enc_obs_list = self.enc_obs(obs_data_list.view(-1, *obs_data_list.shape[2:]))  # ( b * t, d )
        enc_obs_list = enc_obs_list.view(num_samples, full_seq_size, -1)  # ( b, t, d )

        # 観測系列全体から，事後分布 q(m|x) に従って境界フラグをサンプリング
        post_boundary_log_alpha_list = self.post_boundary(enc_obs_list)  # ( b, t, 2 )
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(
            post_boundary_log_alpha_list
        )  # ( b, t, 2 )

        # コンテキスト期間とパディング期間の境界フラグを強制的にREADに設定
        boundary_data_list[:, :(init_size + 1), 0] = 1.0
        boundary_data_list[:, :(init_size + 1), 1] = 0.0
        boundary_data_list[:, -init_size:, 0] = 1.0
        boundary_data_list[:, -init_size:, 1] = 0.0

        # 事後分布 q(z,s|m,x) を計算するための特徴量を，特殊な双方向RNNでエンコード
        abs_post_fwd_list = []
        abs_post_bwd_list = []
        obs_post_fwd_list = []
        abs_post_fwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_post_bwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        obs_post_fwd = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
            # --- 順方向エンコード ---
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)
            # 抽象レベルのRNNは，部分系列の境界に関係なく過去の文脈をすべて蓄積
            abs_post_fwd = self.abs_post_fwd(enc_obs_list[:, fwd_t], abs_post_fwd)
            # 観測レベルのRNNは，境界(READ)で隠れ状態がリセットされる (masked RNN)
            obs_post_fwd = self.obs_post_fwd(
                enc_obs_list[:, fwd_t], fwd_copy_data * obs_post_fwd
            )
            abs_post_fwd_list.append(abs_post_fwd)
            obs_post_fwd_list.append(obs_post_fwd)

            # --- 逆方向エンコード ---
            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            # 抽象レベルの逆方向RNNも，境界(READ)で隠れ状態がリセットされる (masked RNN)
            abs_post_bwd = self.abs_post_bwd(enc_obs_list[:, bwd_t], abs_post_bwd)
            abs_post_bwd_list.append(abs_post_bwd)
            abs_post_bwd = bwd_copy_data * abs_post_bwd
        abs_post_bwd_list = abs_post_bwd_list[::-1]

        # 状態・抽象状態を保持するリストを初期化
        obs_rec_list = []
        prior_abs_state_list = []
        post_abs_state_list = []
        prior_obs_state_list = []
        post_obs_state_list = []
        prior_boundary_log_alpha_list = []

        # 状態と潜在変数を初期化
        abs_belief = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = obs_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = obs_data_list.new_zeros(num_samples, self.obs_state_size)

        # --- 事前分布に従って1ステップずつ遷移 ---
        for t in range(init_size, init_size + seq_size):
            # (0) このステップの境界フラグを取得 (READ or COPY)
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)  # (b, 1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)  # (b, 1)

            # (1) 抽象状態 (z_t) をサンプリング
            # 最初のステップでは，コンテキストから計算した特徴量で信念状態を初期化
            if t == init_size:
                abs_belief = self.init_abs_belief(abs_post_fwd_list[t - 1])
            else:
                # 境界(READ)なら行動で信念を更新，境界でない(COPY)なら維持
                abs_belief = copy_data * abs_belief + read_data * self.update_abs_belief(
                    torch.cat([abs_state, act_data_list[:, t - 1]], dim=1),
                    abs_belief,
                ) # WRITE ME
            # 事前分布 p(z_t|...) を計算
            prior_abs_state = self.prior_abs_state(abs_belief)
            # 事後分布 q(z_t|...) を計算
            post_abs_state = self.post_abs_state(torch.cat(
                [abs_post_fwd_list[t - 1], abs_post_bwd_list[t]], dim=1
            ))
            # 事後分布からサンプリングし，境界に従い状態を更新(UPDATE)または維持(COPY)
            abs_state = copy_data * abs_state + read_data * post_abs_state.rsample() # WRITE ME
            abs_feat = self.abs_feat(torch.cat([abs_belief, abs_state], dim=1))

            # (2) 観測状態 (s_t) をサンプリング
            # 境界(READ)なら抽象特徴で信念を初期化，境界でない(COPY)なら更新
            obs_belief = copy_data * self.update_obs_belief(
                torch.cat([obs_state, abs_feat], dim=1),
                obs_belief,
            ) + read_data * self.init_obs_belief(abs_feat) # WRITE ME
            # 事前分布 p(s_t|...) を計算
            prior_obs_state = self.prior_obs_state(obs_belief)
            # 事後分布 q(s_t|...) を計算
            post_obs_state = self.post_obs_state(torch.cat([obs_post_fwd_list[t], abs_feat], dim=1))
            # 事後分布からサンプリング
            obs_state = post_obs_state.rsample()
            obs_feat = self.obs_feat(torch.cat([obs_belief, obs_state], dim=1))

            # (3) 観測をデコードするための特徴量をリストに保存
            obs_rec_list.append(obs_feat)

            # (4) 境界の事前分布 p(m_t|s_t) を計算
            prior_boundary_log_alpha = self.prior_boundary(obs_feat)

            # (5) 計算結果をリストに保存
            prior_boundary_log_alpha_list.append(prior_boundary_log_alpha)
            prior_abs_state_list.append(prior_abs_state)
            post_abs_state_list.append(post_abs_state)
            prior_obs_state_list.append(prior_obs_state)
            post_obs_state_list.append(post_obs_state)

        # 全ステップの観測を一括でデコード
        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        obs_rec_list = self.dec_obs(obs_rec_list.view(num_samples * seq_size, -1))
        obs_rec_list = obs_rec_list.view(num_samples, seq_size, *obs_rec_list.shape[-3:])

        # 結果をテンソルにまとめる
        prior_boundary_log_alpha_list = torch.stack(prior_boundary_log_alpha_list, dim=1)

        # 損失計算に関係ないパディング部分を削除
        boundary_data_list = boundary_data_list[:, init_size:(init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[:, (init_size + 1):(init_size + 1 + seq_size)]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[:, (init_size + 1):(init_size + 1 + seq_size)]

        # 制約に基づいて事前分布を修正
        prior_boundary_log_alpha_list = self.regularize_prior_boundary(
            prior_boundary_log_alpha_list, boundary_data_list
        )

        # Concrete分布の対数密度を計算
        prior_boundary_log_density = self.log_density_concrete(
            prior_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )
        post_boundary_log_density = self.log_density_concrete(
            post_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )

        # 境界確率を計算
        prior_boundary_list = F.softmax(prior_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        post_boundary_list = F.softmax(post_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        prior_boundary_list = Bernoulli(probs=prior_boundary_list)
        post_boundary_list = Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)

        # 結果をリストで返す
        return [obs_rec_list,
                prior_boundary_log_density,
                post_boundary_log_density,
                prior_abs_state_list,
                post_abs_state_list,
                prior_obs_state_list,
                post_obs_state_list,
                boundary_data_list,
                prior_boundary_list,
                post_boundary_list]

    def jumpy_generation(self, init_data_list, full_action_cond, seq_size):
        """抽象レベルのダイナミクスのみを用い，部分系列単位で未来を生成

        Parameters
        ----------
            init_data_list (torch.Tensor): コンテキストの観測系列．( b, init_size, c, h, w )
            full_action_cond (torch.Tensor): 生成を条件づける行動系列．( b, t, action_size )
            seq_size (int): 生成するジャンプの回数（部分系列の数）
        Returns
        ----------
            torch.Tensor: 生成された観測系列．( b, seq_size, c, h, w )
        """
        assert seq_size <= full_action_cond.shape[1] - init_data_list.shape[1], "Can't generate over the action conditions."

        # 評価モードに設定
        self.eval()

        # 変数を初期化
        num_samples = init_data_list.shape[0]
        init_size = init_data_list.shape[1]

        # コンテキスト期間をエンコード
        abs_post_fwd = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        for t in range(init_size):
            abs_post_fwd = self.abs_post_fwd(self.enc_obs(init_data_list[:, t]), abs_post_fwd)

        # 状態を初期化
        abs_belief = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = init_data_list.new_zeros(num_samples, self.abs_state_size)

        # 結果を保存するリストを初期化
        obs_rec_list = []

        # 抽象レベルで1ステップずつ遷移（ジャンプ）
        for t in range(seq_size):
            # (1) 抽象状態 (z_t) をサンプリング（毎ステップUPDATE）
            if t == 0:
                abs_belief = self.init_abs_belief(abs_post_fwd)
            else:
                abs_belief = self.update_abs_belief(
                    torch.concat([abs_state, full_action_cond[:, init_size + t - 1]], dim=1),
                    abs_belief,
                )
            abs_state = self.prior_abs_state(abs_belief).rsample()
            abs_feat = self.abs_feat(torch.cat([abs_belief, abs_state], dim=1))

            # (2) 観測状態 (s_t) をサンプリング（毎ステップINIT）
            obs_belief = self.init_obs_belief(abs_feat)
            obs_state = self.prior_obs_state(obs_belief).rsample()
            obs_feat = self.obs_feat(torch.cat([obs_belief, obs_state], dim=1))

            # (3) 観測をデコード
            obs_rec = self.dec_obs(obs_feat)

            # (4) 結果を保存
            obs_rec_list.append(obs_rec)

        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        return obs_rec_list, None

    def full_generation(self, init_data_list, full_action_cond, seq_size):
        """学習した階層モデル全体を用いて，1フレームずつ未来を生成

        Parameters
        ----------
            init_data_list (torch.Tensor): コンテキストの観測系列．( b, init_size, c, h, w )
            full_action_cond (torch.Tensor): 生成を条件づける行動系列．( b, t, action_size )
            seq_size (int): 生成するフレーム数
        Returns
        ----------
            torch.Tensor: 生成された観測系列．( b, seq_size, c, h, w )
            torch.Tensor: 生成時にサンプリングされた境界フラグ．( b, seq_size, 1 )
        """
        assert seq_size <= full_action_cond.shape[1] - init_data_list.shape[1], "Can't generate over the action conditions."

        # 評価モードに設定
        self.eval()

        # 変数を初期化
        num_samples = init_data_list.shape[0]
        init_size = init_data_list.shape[1]

        # コンテキスト期間をエンコード
        abs_post_fwd = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        for t in range(init_size):
            abs_post_fwd = self.abs_post_fwd(self.enc_obs(init_data_list[:, t]), abs_post_fwd)

        # 状態を初期化
        abs_belief = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = init_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = init_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = init_data_list.new_zeros(num_samples, self.obs_state_size)

        # 結果を保存するリストを初期化
        obs_rec_list = []
        boundary_data_list = []

        # 1ステップずつ遷移
        read_data = init_data_list.new_ones(num_samples, 1)
        copy_data = 1 - read_data
        for t in range(seq_size):
            # (1) 抽象状態 (z_t) をサンプリング
            if t == 0:
                abs_belief = self.init_abs_belief(abs_post_fwd)
            else:
                abs_belief = read_data * self.update_abs_belief(
                    torch.concat([abs_state, full_action_cond[:, init_size + t - 1]], dim=1),
                    abs_belief,
                ) + copy_data * abs_belief
            abs_state = read_data * self.prior_abs_state(abs_belief).rsample() + copy_data * abs_state
            abs_feat = self.abs_feat(torch.cat([abs_belief, abs_state], dim=1))

            # (2) 観測状態 (s_t) をサンプリング
            obs_belief = read_data * self.init_obs_belief(abs_feat) + \
                copy_data * self.update_obs_belief(torch.cat([obs_state, abs_feat], dim=1), obs_belief)
            obs_state = self.prior_obs_state(obs_belief).rsample()
            obs_feat = self.obs_feat(torch.cat([obs_belief, obs_state], dim=1))

            # (3) 観測をデコード
            obs_rec = self.dec_obs(obs_feat)

            # (4) 結果を保存
            obs_rec_list.append(obs_rec)
            boundary_data_list.append(read_data)

            # (5) 次のステップの境界をサンプリング
            prior_boundary = self.boundary_sampler(self.prior_boundary(obs_feat))[0]
            read_data = prior_boundary[:, 0].unsqueeze(-1)
            copy_data = prior_boundary[:, 1].unsqueeze(-1)

        # 結果をテンソルにまとめる
        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        boundary_data_list = torch.stack(boundary_data_list, dim=1)
        return obs_rec_list, boundary_data_list


class VTA(nn.Module):
    def __init__(
        self,
        belief_size,
        state_size,
        act_size,
        num_layers,
        max_seg_len,
        max_seg_num,
    ):
        super(VTA, self).__init__()

        # --- ネットワークのサイズ定義 ---
        self.belief_size = belief_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        # --- モデルの初期化 ---
        # 階層的状態空間モデル
        self.state_model = HierarchicalRSSM(
            belief_size=self.belief_size,
            state_size=self.state_size,
            act_size=act_size,
            num_layers=self.num_layers,
            max_seg_len=self.max_seg_len,
            max_seg_num=self.max_seg_num,
        )

    def forward(
        self, obs_data_list, act_data_list, seq_size, init_size, obs_std=1.0
    ):
        # (1) 状態空間モデルの順伝播を実行
        [
            obs_rec_list,
            prior_boundary_log_density_list,
            post_boundary_log_density_list,
            prior_abs_state_list,
            post_abs_state_list,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list
        ] = self.state_model(obs_data_list, act_data_list, seq_size, init_size)

        # (2) 再構成誤差を計算 (空間・チャネル次元で合計)
        obs_target_list = obs_data_list[:, init_size:-init_size]
        obs_cost = -Normal(obs_rec_list, obs_std).log_prob(obs_target_list)
        obs_cost = obs_cost.sum(dim=[2, 3, 4])

        # (3) KLダイバージェンスを計算
        # 状態に関するKLダイバージェンス
        kl_abs_state_list = []
        kl_obs_state_list = []
        for t in range(seq_size):
            # READフラグを取得 (COPYの場合はKLを0にするため)
            read_data = boundary_data_list[:, t].detach()

            # KLダイバージェンスを計算 (次元方向に合計)
            kl_abs_state = kl_divergence(post_abs_state_list[t], prior_abs_state_list[t]) * read_data
            kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
            kl_abs_state_list.append(kl_abs_state.sum(-1))
            kl_obs_state_list.append(kl_obs_state.sum(-1))
        kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)
        kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

        # 境界に関するKLダイバージェンス (log q(m|x) - log p(m|s))
        kl_mask_list = (post_boundary_log_density_list - prior_boundary_log_density_list)

        # 結果を辞書で返す
        return {
            'rec_data': obs_rec_list,
            'mask_data': boundary_data_list,
            'obs_cost': obs_cost,
            'kl_abs_state': kl_abs_state_list,
            'kl_obs_state': kl_obs_state_list,
            'kl_mask': kl_mask_list,
            'p_mask': prior_boundary_list.mean,
            'q_mask': post_boundary_list.mean,
            'p_ent': prior_boundary_list.entropy(),
            'q_ent': post_boundary_list.entropy(),
            'beta': self.state_model.mask_beta,
            'train_loss': obs_cost.mean() + kl_abs_state_list.mean() + \
                kl_obs_state_list.mean() + kl_mask_list.mean()
        }

    def jumpy_generation(self, *args):
        return self.state_model.jumpy_generation(*args)

    def full_generation(self, *args):
        return self.state_model.full_generation(*args)