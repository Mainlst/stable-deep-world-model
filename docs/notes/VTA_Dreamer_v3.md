# VTA-DreamerV3の実装について
## 概要

本ドキュメントは、**オリジナルVTA**と**DreamerV3統合版VTA**の両方の仕様をまとめたものです。

## 1. アーキテクチャ概要

### 階層的状態空間モデル

```
┌─────────────────────────────────────────────────────────────────┐
│                    VTA Hierarchical Model                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  Abstract Level │───▶│ Observation Level│                    │
│  │  (z_t, h^z_t)   │    │   ($s_t$, h^s_t)   │                    │
│  └─────────────────┘    └─────────────────┘                     │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│              Boundary Detector (m_t)                            │
│              READ = 新セグメント開始                             │
│              COPY = セグメント継続                               │
└─────────────────────────────────────────────────────────────────┘

```

---

## 2. コンポーネント比較

### 2.1 LatentDistribution

| 項目 | オリジナル | DreamerV3統合版 |
| --- | --- | --- |
| 中間層 | `Linear → ELU` | `Linear → LayerNorm → SiLU` |
| std計算 | `Sigmoid` (範囲: 0-1) | `softplus + min_std` (範囲: 0.1-∞) |
| 戻り値 | `Normal`分布 | `{"mean", "std"}`辞書 |

**オリジナル:**

```python
std = nn.Sequential(nn.Linear(feat_size, latent_size), nn.Sigmoid())
return Normal(loc=self.mean(feat), scale=self.std(feat))

```

**DreamerV3統合版:**

```python
std = F.softplus(self.std_layer(feat)) + self._min_std
return {"mean": mean, "std": std}

```

### 2.2 Encoder / Decoder

**オリジナルVTA（内蔵）:**

```
Encoder: 4層CNN (Conv2d k=4, stride=2) → BatchNorm → ELU → Flatten
Decoder: Linear → 4層TransposedCNN → Tanh (出力範囲: -1〜1)

```

**DreamerV3統合版:**

- Encoder/Decoderは**外部（WorldModel）** で管理
- VTAクラスは`embed`を入力として受け取る

### 2.3 PriorBoundaryDetector

| 項目 | オリジナル | DreamerV3統合版 |
| --- | --- | --- |
| 構造 | `Linear(input, 2)` | `Linear → LayerNorm → SiLU → Linear` |
| 表現力 | 線形決定境界 | 非線形決定境界 |

### 2.4 PostBoundaryDetector

両バージョンとも**1D-Conv**を使用:

```
Conv1d(k=3, padding=1) → BatchNorm → ELU/SiLU → ... → Conv1d(→2)

```

---

## 3. 事後分布の計算（重要な違い）

### オリジナル: 双方向RNN

```python
# 順方向
for fwd_t in range(full_seq_size):
    abs_post_fwd = self.abs_post_fwd(enc_obs[:, fwd_t], abs_post_fwd)
    obs_post_fwd = self.obs_post_fwd(enc_obs[:, fwd_t], copy_data * obs_post_fwd)
# 逆方向
for bwd_t in reversed(range(full_seq_size)):
    abs_post_bwd = self.abs_post_bwd(enc_obs[:, bwd_t], abs_post_bwd)
    abs_post_bwd = copy_data * abs_post_bwd  # 境界でリセット
# 事後分布: 順方向と逆方向を結合
post_abs_state = self.post_abs_state(
    torch.cat([abs_post_fwd_list[t-1], abs_post_bwd_list[t]], dim=1)
)

```

**特徴:**

- 抽象レベル: 順方向は連続蓄積、逆方向は境界でリセット
- 観測レベル: 順方向のみ、境界でリセット
- **未来の情報も考慮**して事後分布を計算

### DreamerV3統合版: シングルパス

```python
# PostBoundaryDetectorで全シーケンスを1D-Convで処理
post_boundary_logits = self.post_boundary(embed)
# Posteriorはbeliefとembedを直接結合
post_abs_input = torch.cat([abs_belief, embed], dim=-1)
post_abs_stats = self.post_abs_state(post_abs_input)

```

**特徴:**

- 双方向RNNを省略し、計算効率を向上
- 1D-Convで時間的コンテキストを近似

---

## 4. 境界サンプリング

### Gumbel-Softmax

```python
def gumbel_sampling(log_alpha, temp, margin=1e-4):
    noise = log_alpha.new_empty(log_alpha.shape).uniform_(margin, 1 - margin)
    gumbel_sample = -torch.log(-torch.log(noise))
    return (log_alpha + gumbel_sample) / temp

```

### Straight-Through Estimator

```python
# 順伝播: one-hot (離散)
sample_data = torch.eye(2)[torch.max(sample_prob, dim=-1)[1]]
# 逆伝播: 連続確率の勾配を伝播
sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())

```

---

## 5. セグメント制約

| パラメータ | デフォルト | 説明 |
| --- | --- | --- |
| `max_seg_len` | 10 | 1セグメントの最大長 |
| `max_seg_num` | 5 | 最大セグメント数 |
| `mask_beta` / `boundary_temp` | 1.0 | 温度パラメータ |
| **優先順位:** `max_seg_len` > `max_seg_num` |  |  |

---

## 6. 損失計算

### オリジナルVTA

```python
# 再構成誤差
obs_cost = -Normal(obs_rec, obs_std).log_prob(obs_target).sum(dim=[2,3,4])
# 状態のKL (抽象レベルは境界時のみ)
kl_abs = kl_divergence(post_abs, prior_abs) * read_data
kl_obs = kl_divergence(post_obs, prior_obs)
# 境界のKL (Concrete分布)
kl_mask = log_density_concrete(post) - log_density_concrete(prior)
# 全損失
train_loss = obs_cost + kl_abs + kl_obs + kl_mask

```

### Concrete分布の対数密度

```python
def log_density_concrete(log_alpha, log_sample, temp):
    exp_term = log_alpha - temp * log_sample
    log_prob = sum(exp_term) - 2.0 * logsumexp(exp_term)
    return log_prob

```

### DreamerV3統合版

損失計算は**WorldModel**で実行:

- 再構成誤差: 各ヘッド（image, reward等）で計算
- KL損失: `VTA.kl_loss()`メソッド
- `dyn_scale`, `rep_scale` による重み付け

---

## 7. 生成モード

| モード | 説明 | 境界 |
| --- | --- | --- |
| `jumpy_generation` | 抽象レベルのみ遷移（各ステップ=1セグメント） | 常にREAD |
| `full_generation` | 両レベル遷移（1ステップ=1フレーム） | Priorからサンプリング |

---

## 8. 状態辞書構造

### オリジナル（内部変数）

```python
abs_belief, abs_state, obs_belief, obs_state
abs_post_fwd_list, abs_post_bwd_list, obs_post_fwd_list

```

### DreamerV3統合版

```python
state = {
    "abs_belief": (B, 512),  "abs_stoch": (B, 32),
    "abs_mean": (B, 32),     "abs_std": (B, 32),
    "obs_belief": (B, 512),  "obs_stoch": (B, 32),
    "obs_mean": (B, 32),     "obs_std": (B, 32),
    "boundary": (B, 1),      "boundary_logit": (B, 2),
    "seg_len": (B, 1),       "seg_num": (B, 1),
}

```

---

## 9. 主要な違いまとめ

| 項目 | オリジナル | DreamerV3統合版 |
| --- | --- | --- |
| Encoder/Decoder | 内蔵 (CNN) | 外部 (WorldModel) |
| Posterior計算 | **双方向RNN** | シングルパス |
| 活性化関数 | ELU | SiLU |
| std計算 | Sigmoid (0-1) | softplus + min_std |
| 正規化 | BatchNorm | LayerNorm |
| 損失計算 | forward内で完結 | 外部 (WorldModel) |
| 境界KL | Concrete分布 | 標準KL |
| DreamerV3互換 | × | ○ |

---

## 参考文献

- Kim, Ahn, Bengio. "Variational Temporal Abstraction" (NeurIPS 2019)
- Hafner et al. "Mastering Diverse Domains through World Models" (DreamerV3, 2023)