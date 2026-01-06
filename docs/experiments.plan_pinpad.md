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
