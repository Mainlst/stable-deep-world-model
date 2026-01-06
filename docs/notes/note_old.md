# 2025-12-23 VTA 決定境界（boundary）学習の調査メモ

## 目的

- Atari（Breakout / Frostbite / PrivateEye）でVTAの時間抽象化（READ/COPY）が「意味のある区切り」で発火できているかを確認し、学習が進まない原因を切り分ける。
- 可視化（境界の発火タイミング、内部量）と、簡易な定量検証（ランダム/周期 vs 意味）まで行う。

## 観測した症状

- **境界が序盤に連続で発火**しているように見えるケース（READが前半に偏る）。
- **まったく発火しない**ように見えるケース（全てCOPYに見える）。
- **10ステップごとに等間隔で発火**しているように見えるケース。

## 主な原因（切り分け結果）

### 1) `vta_max_seg_len` の「強制境界」による周期発火

- `vta.py` の `_regularize_boundary()` は `seg_len >= vta_max_seg_len` で READ を強制する（硬い制約）。
- 境界検出が学習できずCOPYに寄ると、**制約により一定間隔で境界が立つ**（例: `vta_max_seg_len=10` → 10ステップ周期に見える）。

### 2) 「可視化が学習時設定と不一致」だと、周期発火に“見える”

- `scripts/vta_boundary_viz.py` は（以前）学習時のCLIオーバーライドを受け取れず、`configs.yaml` のデフォルトでWMを復元していた。
- その結果、学習時に `vta_max_seg_len=50` を使っていても、可視化側が `10` のまま復元され、**10ステップ周期のように見える**ことがあった。

### 3) 境界が「全発火」「無発火」に崩壊する問題

- 境界は離散（Gumbel-Softmax + straight-through）で、学習初期は不安定になりやすい。
- VTA統合版は、抽象KLを `boundary` で重み付けしているため（`vta.py: kl_loss()`）、境界が極端に偏ると学習信号の分配が歪みやすい。
- さらに、`PostBoundaryDetector` が `embed` だけから境界を推定しているため、学習初期に「簡単な解」に落ちる（ほぼCOPY/ほぼREAD）可能性がある。

## 実施した対策・変更点（コード）

### A) 強制境界の強さを制御できる設定を追加

- `configs.yaml` に `vta_boundary_force_scale` を追加（デフォルト `10.0`、`0.0`で強制を無効化）。
- `vta.py` の `_regularize_boundary()` に反映。
- `models.py` でVTA初期化に `boundary_force_scale` を渡すように変更。

関係ファイル:
- `configs.yaml`
- `vta.py`
- `models.py`

狙い:
- 「学習できないから強制境界で周期的に発火」してしまう状況を切り離し、境界が本当に学習されているか確認しやすくする。

### B) 可視化スクリプトの設定ずれを解消

- `scripts/vta_boundary_viz.py` が `parse_known_args()` で未知引数（学習時オーバーライド）を受け取り、`load_config()` に渡せるように修正。

関係ファイル:
- `scripts/vta_boundary_viz.py`

### C) 境界率の事前分布（ターゲットREAD率）正則化を追加（任意）

- `configs.yaml` に `vta_boundary_rate` / `vta_boundary_scale` を追加（0で無効）。
- `models.py` で、posteriorのREAD確率が目標率に近づくようBernoulli KLを加算し、`boundary_kl` をログ出力。

関係ファイル:
- `configs.yaml`
- `models.py`

狙い:
- 「COPYに潰れて境界が出ない」崩壊を抑制する（ただしスケール調整が必要）。

## 可視化（例）

生成物:
- `boundary_grid.png`: フレーム + 境界バー（READ赤 / COPY青）
- `boundary_internal.png`: READ確率、seg_len/seg_num、abs/obs KL、reward/frame_delta
- `boundary_fired.png`: READ発火フレームだけを時刻付きで並べる

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

## 定量検証（ランダム/周期 vs 意味のある区切り）

### 追加した評価スクリプト

- `scripts/vta_boundary_eval.py` を追加。
- 1エピソードに対して、
  - `frame_delta`（隣接フレーム差分の平均）上位p%イベント
  - `reward != 0`イベント
  を定義し、境界時のイベント率がランダムより高いかを検証する。

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

### 結果（例: `logdir/vta_private_eye_fix3`）

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

## 今後の課題・次の一手

- **報酬が出るまで学習を進めたログ**で、rewardイベントやスコア変化、ライフ減（可能なら）など「意味イベント」を定義して同様に検定する。
- `delta_percentile` を変えて頑健性チェック（95/99など）。
- 境界が COPY に潰れる場合:
  - `vta_boundary_rate` / `vta_boundary_scale` の調整（スケールが強すぎると逆に不安定化するため注意）
  - `vta_boundary_temp`（温度）調整
  - 学習初期だけ強制境界を弱く入れて、その後0にするなどのスケジューリング（未実装）
