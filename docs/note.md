# 2025-01-06 pinpad 検証ログ

## 実行環境
- GPU: RTX 3070
- リポジトリ: /home/user/stable-deep-world-model

## ここまでの要約
- pinpad は Dreamer 側で未対応だったため、pinpad 環境ラッパと suite 追加を行い、RSSM/VTA の学習が開始できる状態にした。
- その後、RTX3070 で軽量寄りの設定に調整しつつ、短時間実行で `metrics.jsonl` が更新される（＝学習ループが回る）ことを確認した。
- 進捗確認のため、`run_pinpad_sweep.sh` に「現在条件/全体」の進捗バー表示とステータスファイル出力を追加し、別プロセスで監視できるようにした。

## 修正内容
- `src_dreamerv3/dreamer.py` に `suite == "pinpad"` を追加。
- `src_dreamerv3/envs/pinpad.py` を追加:
  - `tasks/pinpad` のローカル実装を `sys.path` で読み込み。
  - 観測を Dreamer 互換の Dict 形式に整形（`image`, `is_first`, `is_last`, `is_terminal`）。
  - 離散行動として扱えるよう `action_space.discrete = True` を付与。
- `src_dreamerv3/models.py` を修正:
  - RSSM の `observe()` に `reward=` を渡さない（VTA のみ reward 対応）。
- `src_dreamerv3/configs.yaml` に `pinpad` 設定を追加（`actor.dist=onehot` など）。
- `scripts/run_pinpad_sweep.sh` を改善:
  - 進捗バー（現在条件）+ 全体 `(run_index/total_runs)` 表示。
  - `RUN_LOG` へ `tee` でターミナル/ログ両方に出力。
  - `STATUS_FILE` に現在進捗を随時書き出し。
- `scripts/monitor_pinpad_sweep.sh` を追加:
  - `STATUS_FILE` を読み取り、進捗バー形式で常時表示する監視スクリプト。

## 実行コマンド
```bash
TASK=pinpad_eight bash scripts/run_pinpad_sweep.sh
LOGROOT=logdir/pinpad_fix2 TASK=pinpad_eight bash scripts/run_pinpad_sweep.sh
LOGROOT=logdir/pinpad_fast1 TASK=pinpad_eight bash scripts/run_pinpad_sweep.sh
LOGROOT=logdir/pinpad_fast2 TASK=pinpad_eight bash scripts/run_pinpad_sweep.sh
python3 -m src_dreamerv3.dreamer --logdir logdir/pinpad_vta_start1/pinpad_vta_force0_t1_seed0 --configs pinpad --task pinpad_eight --steps 40000 --eval_every 10000 --log_every 10000 --envs 1 --prefill 1200 --batch_size 12 --batch_length 40 --train_ratio 320 --size 64 --action_repeat 1 --time_limit 1000 --precision 16 --compile True --seed 0 --dynamics_type vta --vta_max_seg_len 30 --vta_boundary_force_scale 0.0 --vta_boundary_temp 1.0 --vta_boundary_rate 0.0 --vta_boundary_scale 0.0
LOGROOT=logdir/pinpad_full1 TASK=pinpad_eight nohup bash scripts/run_pinpad_sweep.sh > logdir/pinpad_full1/run.log 2>&1 &
LOGROOT=logdir/pinpad_full3 TASK=pinpad_eight RUN_LOG=logdir/pinpad_full3/pinpad_sweep.log STATUS_FILE=logdir/pinpad_full3/pinpad_status.env nohup bash scripts/run_pinpad_sweep.sh > logdir/pinpad_full3/run.log 2>&1 &
STATUS_FILE=logdir/pinpad_full3/pinpad_status.env bash scripts/monitor_pinpad_sweep.sh
```

## 結果
- 失敗: Dreamer の環境生成で `NotImplementedError: pinpad`
- ログ作成先: `logdir/pinpad_rssm_seed0/config.yaml` まで作成されたが、環境作成で停止
-
- 修正後: `LOGROOT=logdir/pinpad_fix2` で RSSM の学習が開始し、`metrics.jsonl` に学習ログが記録されることを確認。
- 120秒の待機後もプロセスは継続中 (タイムアウトで終了)。
- ログ: `logdir/pinpad_fix2/pinpad_rssm_seed0/metrics.jsonl`
  - 例: step 1000 時点で `model_loss` などが記録されている。
-
- 高速化設定の試行 (`logdir/pinpad_fast1`) は負荷が高く、手動中断。
-
- 軽量化設定 (`logdir/pinpad_fast2`) で 120 秒実行し、`metrics.jsonl` に学習ログが記録されることを確認。
  - 例: step 1200 で `model_loss` などが更新されている。
-
- VTA 開始 (`logdir/pinpad_vta_start1/pinpad_vta_force0_t1_seed0`) を 120 秒実行。
  - `metrics.jsonl` に step 1200 の更新を確認。
  - `boundary_ratio` がログに出ており、境界学習が動作している。
-
- 一通り検証を再開: `logdir/pinpad_full1/run.log` に出力しながら `run_pinpad_sweep.sh` をバックグラウンド実行中。
  - PID: 16058

## 現在の進め方（再現可能な手順）
### 1) 一括スイープをバックグラウンドで回す
- 推奨: `LOGROOT` を毎回変える（途中停止や再実行でログが混ざりにくい）。
- `RUN_LOG` で詳細ログを保存しつつ、`STATUS_FILE` で進捗を機械可読に出す。

例:
```bash
LOGROOT=logdir/pinpad_full3 \
TASK=pinpad_eight \
RUN_LOG=logdir/pinpad_full3/pinpad_sweep.log \
STATUS_FILE=logdir/pinpad_full3/pinpad_status.env \
nohup bash scripts/run_pinpad_sweep.sh > logdir/pinpad_full3/run.log 2>&1 &
```

### 2) 進捗バーで監視する（現在条件＋全体進捗）
```bash
STATUS_FILE=logdir/pinpad_full3/pinpad_status.env bash scripts/monitor_pinpad_sweep.sh
```

### 3) 停止する
```bash
pgrep -fl "scripts/run_pinpad_sweep.sh|src_dreamerv3.dreamer"
kill <PID...>
```

## 結果整理（現時点）
### 実行セット: `logdir/pinpad_full3`
- スイープ構成: `SEEDS=0` のため全4条件（RSSM + VTA 3条件）を想定（`(1/4)〜(4/4)` 表示）。
- 進捗/ログ:
  - 集約ログ: `logdir/pinpad_full3/pinpad_sweep.log`
  - 端末ログ: `logdir/pinpad_full3/run.log`
  - 進捗状態: `logdir/pinpad_full3/pinpad_status.env`

### (1/4) RSSM ベースライン: `logdir/pinpad_full3/pinpad_rssm_seed0`
- 状態: 完走（`latest.pt` 生成あり）。
- 代表ログ:
  - `logdir/pinpad_full3/pinpad_rssm_seed0/metrics.jsonl`
  - `logdir/pinpad_full3/pinpad_rssm_seed0/latest.pt`
- 観測:
  - `train_return` は 0.0 のまま推移（少なくともこの設定/期間では報酬が発生していない可能性）。
  - `fps` はログ上で約 5.3〜5.4。

### (2/4) VTA (force=0, temp=1): `logdir/pinpad_full3/pinpad_vta_force0_t1_seed0`
- 状態: 早期停止（CUDA device-side assert）。
- 代表ログ:
  - `logdir/pinpad_full3/pinpad_vta_force0_t1_seed0/metrics.jsonl`（step 1200 まで）
  - `logdir/pinpad_full3/pinpad_sweep.log`（スタックトレースあり）
- 観測:
  - step 1200 で `boundary_ratio ≈ 0.39` が出ており、境界ヘッド自体は動作している。
  - ただし `model_grad_norm=NaN` が出ており不安定。
  - エラー: `probability tensor contains either inf/nan or element < 0` → `torch.AcceleratorError: CUDA error: device-side assert triggered`
- 影響:
  - `latest.pt` が生成されず、後続の `vta_boundary_viz.py` が `Missing checkpoint .../latest.pt` で失敗。

### (3/4) / (4/4)
- 状態: (2/4) のクラッシュにより未実行（後続条件に進めていない）。

### 実行セット: `logdir/pinpad_full10`（修正後の再実行）
- 目的: VTA 条件が落ちずに学習継続し、チェックポイント生成→可視化/定量評価まで到達すること。
- 実行条件（要点）:
  - `SKIP_RSSM=1`（RSSM ベースラインは `pinpad_full3` で完走済みのため省略）
  - VTA は安定化のため `--precision 32` + `--compile False` + `--eval_every 2000`（= 早めに `latest.pt` を作る）
  - `PRETRAIN=10`（初動の学習負荷を軽く）
- 状態（確認時点）:
  - `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/latest.pt` 生成を確認（= 学習が継続して checkpoint に到達）
  - `logdir/pinpad_full10/pinpad_status.env` で進捗を確認可能

#### (VTA) force=0, temp=1: `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0`
- 可視化出力:
  - `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/boundary_grid.png`
  - `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/boundary_internal.png`
  - `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/boundary_fired.png`
- 定量評価（1 episode, permutations=50, delta_percentile=90）:
  - 出力: `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/boundary_eval.json`
  - `reward` は 0 のため評価不能（event_rate=0）
  - `delta` は境界に偏っており、ランダムより有意:
    - `event_at_boundary`: 0.877
    - `lift_over_event_rate`: 7.70
    - `perm_p_value`: 0.0196
    - `periodic_baseline`: 0.125

## 修正（pinpad_full10 再実行に反映）
- VTA の安定化:
  - `src_dreamerv3/networks.py`: `OneHotDist` へ渡す logits を `nan_to_num + clamp`。
  - `src_dreamerv3/vta.py`: `LatentDistribution.get_dist()` で mean/std を `nan_to_num + clamp`。
- スイープの継続性/観測性:
  - `scripts/run_pinpad_sweep.sh`: VTA で `--precision 32`, `--compile False`, `--eval_every 2000` を上書き。
  - `scripts/run_pinpad_sweep.sh`: 失敗してもスイープを継続し、`latest.pt` が無い場合は viz/eval をスキップ。

## 最終結果まとめ（pinpad / VTA 境界検出）
### 完了状況
- 実行: `logdir/pinpad_full10`
- `scripts/run_pinpad_sweep.sh` は VTA 3条件を完走（`SKIP_RSSM=1` のため VTA のみ）。
- 実行ログ（開始/完了時刻）:
  - `logdir/pinpad_full10/pinpad_sweep.log`
    - `pinpad_vta_force0_t1_seed0`: 2026-01-06 09:46:58 → 13:24:06
    - `pinpad_vta_force10_t1_seed0`: 2026-01-06 13:24:19 → 17:18:54
    - `pinpad_vta_force0_t2_rate_seed0`: 2026-01-06 17:19:09 → 20:57:31

### 共通観測
- `reward` は全条件で 0（`boundary_eval.json` の reward event_rate=0）→ 報酬イベントとの相関評価は不可。
- 評価は `frame_delta`（隣接フレーム差分上位10%）イベントで実施（`delta_percentile=90`）。

### 条件別まとめ（deltaイベント）
#### 1) `pinpad_vta_force0_t1_seed0`（`max_seg_len=30, force_scale=0, temp=1`）
- 成果物:
  - checkpoint: `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/latest.pt`
  - 可視化: `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/boundary_grid.png`
  - 定量: `logdir/pinpad_full10/pinpad_vta_force0_t1_seed0/boundary_eval.json`
- 指標（1 episode, permutations=50）:
  - boundary_count: 57
  - event_at_boundary: 0.877
  - lift_over_event_rate: 7.70
  - perm_p_value: 0.0196（有意）
  - periodic_baseline: 0.125
  - auc_read_prob: 0.692
- 解釈: 少なくともこの条件では、境界が「画面が大きく変わるタイミング」に偏って出ている兆候がある。

#### 2) `pinpad_vta_force10_t1_seed0`（`max_seg_len=30, force_scale=10, temp=1`）
- 成果物:
  - checkpoint: `logdir/pinpad_full10/pinpad_vta_force10_t1_seed0/latest.pt`
  - 可視化: `logdir/pinpad_full10/pinpad_vta_force10_t1_seed0/boundary_grid.png`
  - 定量: `logdir/pinpad_full10/pinpad_vta_force10_t1_seed0/boundary_eval.json`
- 指標:
  - boundary_count: 33
  - event_at_boundary: 0.0606
  - lift_over_event_rate: 0.485
  - perm_p_value: 0.941（有意差なし）
  - periodic_baseline: 0.0606
  - auc_read_prob: 0.465
- 解釈: 強制境界が強い条件では、frame_delta との整合が弱く、境界が「意味イベント」に寄っていない可能性。

#### 3) `pinpad_vta_force0_t2_rate_seed0`（`max_seg_len=50, force_scale=0, temp=2, rate/scale=0.1/0.5`）
- 成果物:
  - checkpoint: `logdir/pinpad_full10/pinpad_vta_force0_t2_rate_seed0/latest.pt`
  - 可視化: `logdir/pinpad_full10/pinpad_vta_force0_t2_rate_seed0/boundary_grid.png`
  - 定量: `logdir/pinpad_full10/pinpad_vta_force0_t2_rate_seed0/boundary_eval.json`
- 指標:
  - boundary_count: 0（境界が無発火）
  - delta 評価は NaN（境界が無いため）
  - auc_read_prob: 0.559
- 解釈: この `rate/scale` 設定では境界が COPY に潰れている可能性が高い。

## 修正内容
- `src_dreamerv3/dreamer.py` に pinpad suite を追加。
- `src_dreamerv3/envs/pinpad.py` を作成し、`is_first/is_last/is_terminal` を付与。
- RSSM で `reward` 引数を渡さないように `src_dreamerv3/models.py` を修正。
- `src_dreamerv3/configs.yaml` に `pinpad` 設定を追加し、離散行動に対応。

## エラーログ (抜粋)
```
NotImplementedError: pinpad
```

## 次のアクション候補
- `src_dreamerv3/dreamer.py` の `make_env()` に `suite == "pinpad"` を追加する。
- pinpad 環境の観測を `{"image": ...}` に整形するラッパを用意する。
- 既存のログがある場合は `LOGROOT` を変えて新規実行する。
