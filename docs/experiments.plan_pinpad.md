# VTA+Dreamer / tasks/pinpad 境界検出 実験計画 (RTX 3070)

## 目的
- tasks/pinpad 環境で VTA の境界 (READ/COPY) が「意味のある区切り」に対応できるかを確認する。
- 本番学習の前に、短時間の試行で「うまくいきそうな兆候」を探し、条件を絞り込む。

## 参考メモからの注意点 (docs/notes 準拠)
- `vta_max_seg_len` の強制境界で周期発火に見える可能性があるため、`vta_boundary_force_scale=0.0` の比較が必須。
- 可視化側が学習時オーバーライドと一致していることを確認する。
- 境界が全発火/無発火に崩壊しやすいので、`vta_boundary_temp` と `vta_boundary_rate/scale` を段階調整する。

## 成功指標 (兆候)
- 境界発火が「全発火」「無発火」「等間隔周期」に偏らず、シーン変化や報酬イベントと関係する。
- `vta_boundary_eval.py` で `lift_over_event_rate > 2.0` かつ `perm_p_value < 0.05` が出る。
- `vta_boundary_viz.py` で READ が短い局所で固まりすぎず、COPY と混在する。

## 事前準備 (最初の1日)
- pinpad 環境の観測形式確認 (画像サイズと dtype)。必要なら `src_dreamerv3/envs/pinpad.py` を作り、観測を `{"image": ...}` に整形。
- `src_dreamerv3/dreamer.py` の `make_env()` に `suite == "pinpad"` を追加する計画を立てる。
- まず RSSM (VTAなし) で学習が回ることを確認する。

## 予備実験: 兆候探索フェーズ (短時間で多数試す)
RTX 3070 で 1 実験 30-90 分を目安に縮小設定で回す。

### フェーズA: RSSM ベースライン確認 (VTAなし)
- 目的: 環境/ログ/報酬の有無を確認。
- 設定: `dynamics_type=rssm`、`steps=2e4-5e4`、`envs=1`。
- 観測: `train_return` と `train_openl` を確認。

### フェーズB: VTA 最小構成の挙動確認
- 目的: 境界が学習可能かを短期で見る。
- 設定例 (共通): `dynamics_type=vta`、`steps=2e4-5e4`、`envs=1`。
- 変数:
  - `vta_max_seg_len`: 30 / 50
  - `vta_boundary_force_scale`: 0.0 / 10.0
  - `vta_boundary_temp`: 1.0 / 2.0
  - `vta_boundary_rate/scale`: 0.0 / (0.1, 0.5)
- 出力評価:
  - `scripts/dreamerv3/vta_boundary_viz.py` で境界の可視化 (短いエピソード長で十分)。
  - `scripts/dreamerv3/vta_boundary_eval.py` で `frame_delta` と `reward != 0` の一致率を確認。

### フェーズC: 兆候が出た条件の絞り込み
- 兆候が出た設定を 2-3 seed で再確認。
- 境界の「崩壊」や「強制境界由来の周期性」がないかを必ず検証。

## 本番学習フェーズ (兆候が出た条件のみ)
- 目的: 境界が安定して意味イベントに対応するか検証。
- 設定例: `steps=2e5-5e5`、`envs=1`、`eval_every=1e4`。
- 2-3 seed で再現性を確認。

## RTX 3070 向けリソース調整 (推奨)
- `batch_size`: 8-16、`batch_length`: 32-64 (OOM なら 8/32 まで下げる)
- `precision`: 16 (不安定なら 32 に戻す)
- `compile`: True で不具合が出る場合は False
- `prefill`: 1000-2500 (まず小さめ)
- `size`: 64x64 (pinpad が小さい場合も固定)
- `train_ratio`: 256-512

## 記録と判定
- ログ保存: `logdir` を実験ごとに分ける (例: `logdir/pinpad_vta_sweep_*`)。
- 成否判定:
  - 境界が意味イベント (報酬 or frame_delta) と有意に一致
  - 境界が「周期」「全発火」「無発火」から脱却

## 次のアクション候補 (兆候が弱い場合)
- `vta_boundary_rate/scale` を弱く入れる (COPY 崩壊対策)。
- `vta_boundary_temp` を上げて探索性を上げる。
- `vta_max_seg_len` を長くして強制境界の影響を減らす。

---

## フォローアップ実験（中間報告に向けて「有効性」を強くする）

現状の結論は「pinpad で VTA 境界が `frame_delta` イベントに有意に整合する条件が1つ見つかった」が、seed とイベント定義（reward）が弱点。
来週の中間報告では、以下を最小コストで固めて“主張の強さ”を上げる。

### 目標（中間報告で示したいこと）
- **再現性**: 当たり条件で seed を変えても同様の傾向が出る（偶然ではない）。
- **失敗も含む制御可能性**: `force_scale` や `temp` で崩壊/不整合が起きることも示し、「ハイパラで挙動が変わる＝手法が効いている」ことを説明できる。
- **意味イベント**: 可能なら `reward != 0`（または pinpad固有の意味イベント）で評価し、`frame_delta` だけの主張にならないようにする。

### 成功条件（最低ライン）
- seed 3 本（例: 0/1/2）で、当たり条件が **2/3 以上**で `lift_over_event_rate > 2.0` かつ `perm_p_value < 0.05`（`frame_delta` イベント）
- `periodic_baseline` が極端に高くない（例: 0.2 未満を目安。高い場合は周期性の疑いとして追加検証へ）

### P0: ブロッカー解消（reward が常に 0 の切り分け）
- 目的: 「報酬イベントに境界が寄る」評価を可能にする（最も分かりやすい主張）。
- 手順:
  - 1 episode の素の `reward` シーケンスを確認（環境仕様として 0 なのか、ラッパで落としているのかを切り分け）。
  - 仕様として 0 が正しい場合でも、pinpad 固有の **意味イベント**（例: 内部状態遷移、キー取得、ゴール到達、スコア変化相当）を定義し、`vta_boundary_eval.py` のイベントとして追加する。
- 判定: `reward != 0` の event_rate が 0 ではなくなる（or 代替イベントが取れる）。

### P1: 再現性（当たり条件の seed 追加）
- 対象条件（現状の当たり）:
  - `max_seg_len=30, force_scale=0, temp=1, rate/scale=0`
- 実行: seed を 0/1/2（最低でも追加で 2 本）
- 収集: 各 seed で
  - `boundary_eval.json`（lift/p/periodic_baseline/boundary_count）
  - `boundary_grid.png` / `boundary_fired.png`（中間報告でそのまま貼れる）

### P2: 周期性の分離（強制境界の寄与を切る）
- 目的: 「当たりが `max_seg_len` 強制境界由来の周期」ではないことを確認。
- 実験:
  - 当たり条件のまま `max_seg_len` を 30→50（または 80）に変更して 1 seed だけ実施
- 判定:
  - `periodic_baseline` が上がり過ぎない、かつ lift/p が極端に崩れないなら「強制境界だけではない」材料になる。

### P3: 最小追加スイープ（崩壊を避けつつ探索）
現状の観測では `force=10` は不整合、`temp=2 + rate/scale=0.1/0.5` は無発火（崩壊）。
中間報告前の追加探索は、以下の“中間値”だけに絞る（やり過ぎない）。

- `force_scale` を弱く入れる:
  - `force_scale=1`（`temp=1, max_seg_len=30, rate/scale=0`）
- `temp` を中間にする:
  - `temp=1.5`（`force=0, max_seg_len=30, rate/scale=0`）
- `rate/scale` を小さくする（COPY 側崩壊/無発火の回避）:
  - 例: `rate=0.02, scale=0.1`（`force=0, temp=1.5, max_seg_len=50` など）

### P4: 中間報告用の成果物チェックリスト
- 1枚表（条件×seed）: `lift`, `p`, `boundary_count`, `periodic_baseline`, `auc_read_prob` を並べる
- 代表例の図: 当たり条件の `boundary_grid.png` と `boundary_fired.png` を 1-2枚
- 失敗例の図: 無発火（boundary_count=0）と不整合（lift≪1）の可視化を各1枚
- 研究メッセージ: 「(i) 境界は学習でき、(ii) ある条件では意味イベントに寄り、(iii) 条件で崩壊もする＝制御が必要」

---

## 今回の実験まとめ（2026-01-06 / logdir/pinpad_full10）
対象: VTA 3条件（RSSM は SKIP）。評価は `vta_boundary_eval.py` の `frame_delta`（上位10%）イベント。

### 成功指標に照らした判定
- 報酬イベント（`reward != 0`）: 全条件で `reward` が 0 のため、報酬との相関は評価不能（イベント発生率 0）。
- frame_delta との一致（`lift_over_event_rate > 2.0` かつ `perm_p_value < 0.05`）:
  - ✅ `pinpad_vta_force0_t1_seed0`（max_seg_len=30, force=0, temp=1）
    - lift=7.70, p=0.0196（有意）
    - periodic_baseline=0.125（境界が完全周期ではないが、周期寄りの可能性は引き続き要確認）
  - ❌ `pinpad_vta_force10_t1_seed0`（max_seg_len=30, force=10, temp=1）
    - lift=0.485, p=0.941（有意差なし）
  - ❌ `pinpad_vta_force0_t2_rate_seed0`（max_seg_len=50, force=0, temp=2, rate/scale=0.1/0.5）
    - boundary_count=0（無発火で崩壊）

### 追加観測（崩壊/周期性の観点）
- 強制境界を強くすると（force=10）、frame_delta と整合しない方向に寄る可能性。
- rate/scale を入れて temp を上げた条件は「無発火」側に倒れた（COPY 崩壊の反対側＝READ 崩壊に近い挙動）。

## 今後の優先タスク
### P0: ブロッカー（報酬が常に0）
- pinpad 環境で `reward` が本当に 0 なのか切り分け（環境仕様 vs ラッパの取りこぼし）。
  - まずは 1 episode を手動ロールアウトして、素の reward シーケンスを確認する。
  - 報酬が本来出る設計なら、`src_dreamerv3/envs/pinpad.py` の `step()` で reward を落としていないか確認する。

### P1: 兆候が出た条件の再現性
- `pinpad_vta_force0_t1`（force=0,temp=1,max_seg_len=30）を seed 2-3 本で再実行し、lift/p 値の再現性を見る。
- 同条件で `max_seg_len` だけ 30→50 にして、周期性（強制境界）の寄与を分離する。

### P2: ハイパラの次候補（崩壊回避しつつ探索）
- `force_scale` は 0 を基準に、入れるなら 0→1→2 程度の弱い値から（10 は強すぎる可能性）。
- `temp` は 1.0 を基準に、上げる場合は 1.5 程度の中間を挟む（2.0 + rate/scale は無発火に倒れた）。
- `rate/scale` を入れる場合は、まず小さめ（例: rate=0.02, scale=0.1）から段階的に（今回の 0.1/0.5 は強かった可能性）。

### P3: 判定の強化（frame_delta 以外も含める）
- pinpad の「意味イベント」を追加定義できるなら（例: スコア変化、鍵取得、ゴール到達など）、`vta_boundary_eval.py` にイベントを追加して評価を多面的にする。
- `vta_boundary_viz.py` の READ/COPY の局所集中（短い塊・長い塊）を定性的にチェックし、指標（例: セグメント長分布）を併記できると判断が早い。
