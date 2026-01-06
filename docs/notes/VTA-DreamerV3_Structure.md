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

---

# **2025-12-23 VTA 決定境界（boundary）学習の調査メモ**

## **目的**

- Atari（Breakout / Frostbite / PrivateEye）でVTAの時間抽象化（READ/COPY）が「意味のある区切り」で発火できているかを確認し、学習が進まない原因を切り分ける。
- 可視化（境界の発火タイミング、内部量）と、簡易な定量検証（ランダム/周期 vs 意味）まで行う。

## **観測した症状**

- **境界が序盤に連続で発火**しているように見えるケース（READが前半に偏る）。
- **まったく発火しない**ように見えるケース（全てCOPYに見える）。
- **10ステップごとに等間隔で発火**しているように見えるケース。

## **主な原因（切り分け結果）**

### **1) `vta_max_seg_len` の「強制境界」による周期発火**

- `vta.py` の `_regularize_boundary()` は `seg_len >= vta_max_seg_len` で READ を強制する（硬い制約）。
- 境界検出が学習できずCOPYに寄ると、**制約により一定間隔で境界が立つ**（例: `vta_max_seg_len=10` → 10ステップ周期に見える）。

### **2) 「可視化が学習時設定と不一致」だと、周期発火に“見える”**

- `scripts/vta_boundary_viz.py` は（以前）学習時のCLIオーバーライドを受け取れず、`configs.yaml` のデフォルトでWMを復元していた。
- その結果、学習時に `vta_max_seg_len=50` を使っていても、可視化側が `10` のまま復元され、**10ステップ周期のように見える**ことがあった。

### **3) 境界が「全発火」「無発火」に崩壊する問題**

- 境界は離散（Gumbel-Softmax + straight-through）で、学習初期は不安定になりやすい。
- VTA統合版は、抽象KLを `boundary` で重み付けしているため（`vta.py: kl_loss()`）、境界が極端に偏ると学習信号の分配が歪みやすい。
- さらに、`PostBoundaryDetector` が `embed` だけから境界を推定しているため、学習初期に「簡単な解」に落ちる（ほぼCOPY/ほぼREAD）可能性がある。

## **実施した対策・変更点（コード）**

### **A) 強制境界の強さを制御できる設定を追加**

- `configs.yaml` に `vta_boundary_force_scale` を追加（デフォルト `10.0`、`0.0`で強制を無効化）。
- `vta.py` の `_regularize_boundary()` に反映。
- `models.py` でVTA初期化に `boundary_force_scale` を渡すように変更。

関係ファイル:

- `configs.yaml`
- `vta.py`
- `models.py`

狙い:

- 「学習できないから強制境界で周期的に発火」してしまう状況を切り離し、境界が本当に学習されているか確認しやすくする。

### **B) 可視化スクリプトの設定ずれを解消**

- `scripts/vta_boundary_viz.py` が `parse_known_args()` で未知引数（学習時オーバーライド）を受け取り、`load_config()` に渡せるように修正。

関係ファイル:

- `scripts/vta_boundary_viz.py`

### **C) 境界率の事前分布（ターゲットREAD率）正則化を追加（任意）**

- `configs.yaml` に `vta_boundary_rate` / `vta_boundary_scale` を追加（0で無効）。
- `models.py` で、posteriorのREAD確率が目標率に近づくようBernoulli KLを加算し、`boundary_kl` をログ出力。

関係ファイル:

- `configs.yaml`
- `models.py`

狙い:

- 「COPYに潰れて境界が出ない」崩壊を抑制する（ただしスケール調整が必要）。

## **可視化（例）**

生成物:

- `boundary_grid.png`: フレーム + 境界バー（READ赤 / COPY青）
    
    ![boundary_grid.png](attachment:dbb9e12e-6e58-4afe-8171-cb9df61ae690:boundary_grid.png)
    
- `boundary_internal.png`: READ確率、seg_len/seg_num、abs/obs KL、reward/frame_delta
    
    ![boundary_internal.png](attachment:1c268503-c2bd-40fd-b004-a5d3d754510d:boundary_internal.png)
    
- `boundary_fired.png`: READ発火フレームだけを時刻付きで並べる
    
    ![boundary_fired.png](attachment:10767a98-d559-4da9-b323-e816b1994aec:boundary_fired.png)
    

実行例:

```bash
python3 scripts/vta_boundary_viz.py \
  --logdir logdir/vta_private_eye_fix3 \
  --task atari_private_eye \
  --configs atari100k \
  --length 30 \
  --window reward_or_delta \
  --vta_max_seg_len 50 --vta_max_seg_num 200 --vta_boundary_temp 2.0 --vta_boundary_force_scale 0.0

```

## **定量検証（ランダム/周期 vs 意味のある区切り）**

### **追加した評価スクリプト**

- `scripts/vta_boundary_eval.py` を追加。
- 1エピソードに対して、
    - `frame_delta`（隣接フレーム差分の平均）上位p%イベント
    - `reward != 0`イベント を定義し、境界時のイベント率がランダムより高いかを検証する。

出力指標:

- `event_at_boundary` / `event_at_nonboundary` / `lift_over_event_rate`
- `perm_p_value`: 境界数固定で境界位置をランダムに置き換えるPermutation test
- `periodic_baseline`: 同じ境界数で等間隔に置いたときのイベント率
- `auc_read_prob`: READ確率がイベントをどれだけ順位付けできるか（AUC）

実行例:

```bash
python3 scripts/vta_boundary_eval.py \
  --logdir logdir/vta_private_eye_fix3 \
  --task atari_private_eye \
  --configs atari100k \
  --episodes 3 \
  --delta_percentile 90 \
  --permutations 200 \
  --out_json logdir/vta_private_eye_fix3/boundary_eval.json \
  --vta_max_seg_len 50 --vta_max_seg_num 200 --vta_boundary_temp 2.0 --vta_boundary_force_scale 0.0

```

### **結果（例: `logdir/vta_private_eye_fix3`）**

- `reward` は（今回の3エピソードでは）常に0で、**報酬イベントとの相関は評価不能**。
- `frame_delta` 上位10%イベントは、境界時に高確率で一致:
    - `event_at_boundary`: 0.75〜0.94
    - `lift_over_event_rate`: 7.4〜9.3倍
    - `perm_p_value`: 約0.005（ランダム配置より有意）
    - `periodic_baseline`: 0.0〜0.125（等間隔より高い）
    - Cohen’s d（境界時のdeltaが大きい）: 1.65〜2.82

解釈:

- 少なくともこの条件では、境界は「完全ランダム」ではなく、**画面が大きく変わるタイミング（scene change）に偏って出ている**。
- ただし、**タスク上の意味（報酬/ライフ減など）に対応しているか**は、このログでは未確定。

## **今後の課題・次の一手**

- **報酬が出るまで学習を進めたログ**で、rewardイベントやスコア変化、ライフ減（可能なら）など「意味イベント」を定義して同様に検定する。
- `delta_percentile` を変えて頑健性チェック（95/99など）。
- 境界が COPY に潰れる場合:
    - `vta_boundary_rate` / `vta_boundary_scale` の調整（スケールが強すぎると逆に不安定化するため注意）
    - `vta_boundary_temp`（温度）調整
    - 学習初期だけ強制境界を弱く入れて、その後0にするなどのスケジューリング（未実装）