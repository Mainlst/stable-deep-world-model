# クラウド再現手順（GPU VM向け）

このドキュメントは、本リポジトリ（`stable-deep-world-model`）の実験をクラウド上で**なるべく再現性高く**実行するための手順書です。
対象は主に `src_dreamerv3`（VTA+DreamerV3）と `tasks/pinpad` の実験です。

---

## 0. 前提（重要）

- GPU が見えること（`nvidia-smi` が成功すること）。
- OS は Ubuntu 22.04 を想定（他でも概ね同様）。
- 実験の実行方法は次のいずれか:
  - 直接実行: `python3 -m src_dreamerv3.dreamer ...`
  - 一括スイープ: `bash scripts/run_pinpad_sweep.sh`
- pinpad の取り込みは 2 通り:
  - **推奨（再現性/可搬性が高い）**: `pip install -e tasks/pinpad`
  - 代替: リポジトリ内のパスを参照（本コードでは `src_dreamerv3/envs/pinpad.py` が `tasks/pinpad/src` を `sys.path` に追加するため、追加作業なしでも動く）

---

## 1. GPU VM の準備

### 1.1 GPU確認
```bash
nvidia-smi
```

### 1.2 依存（最低限）
```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
```

---

## 2. リポジトリ取得

```bash
git clone <YOUR_REPO_URL>
cd stable-deep-world-model
```

（任意）実験の完全再現のため、コミットを固定します:
```bash
git rev-parse HEAD
```

---

## 3. Python環境（venv）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
```

---

## 4. 依存インストール（DreamerV3系）

DreamerV3統合用の依存は `requirements-dreamerv3.txt` にピン留めされています。

```bash
pip install -r requirements-dreamerv3.txt
```

（任意）VTA/Balls 側の依存も含めたい場合:
```bash
pip install -r requirements.txt
```

---

## 5. pinpad の導入（推奨）

```bash
pip install -e tasks/pinpad
python -c "from pinpad import PinPad; env=PinPad.make(layout='eight'); env.reset(); print('ok')"
```

※ `layout` は `three/four/five/six/seven/eight` が有効です。

---

## 6. 実行（pinpad / VTA+Dreamer）

### 6.1 進捗監視つきスイープ（推奨）

pinpad の VTA 条件を順次実行し、進捗バーとログを残します。

```bash
LOGROOT=logdir/pinpad_cloud1 \
TASK=pinpad_eight \
SKIP_RSSM=1 \
PRETRAIN=10 \
VTA_EVAL_EVERY=2000 \
RUN_LOG=logdir/pinpad_cloud1/pinpad_sweep.log \
STATUS_FILE=logdir/pinpad_cloud1/pinpad_status.env \
nohup bash scripts/run_pinpad_sweep.sh > logdir/pinpad_cloud1/run.log 2>&1 &
```

別ターミナルで進捗バー監視:
```bash
STATUS_FILE=logdir/pinpad_cloud1/pinpad_status.env bash scripts/monitor_pinpad_sweep.sh
```

ログ確認:
```bash
tail -n 200 logdir/pinpad_cloud1/pinpad_sweep.log
tail -n 200 logdir/pinpad_cloud1/run.log
```

### 6.2 単発実行（デバッグ向け）
```bash
python3 -m src_dreamerv3.dreamer \
  --logdir logdir/pinpad_single \
  --configs pinpad \
  --task pinpad_eight \
  --dynamics_type vta \
  --steps 40000 \
  --eval_every 2000 --log_every 2000 \
  --prefill 1200 --pretrain 10 \
  --batch_size 12 --batch_length 40 --train_ratio 320 \
  --compile False --precision 32
```

---

## 7. 生成物と評価（境界の可視化/定量）

スイープ成功時、各実験ディレクトリに:
- `latest.pt`（チェックポイント）
- `metrics.jsonl`
- `boundary_grid.png` / `boundary_internal.png` / `boundary_fired.png`
- `boundary_eval.json`

手動評価コマンド例:
```bash
python3 scripts/dreamerv3/vta_boundary_viz.py \
  --logdir logdir/pinpad_cloud1/pinpad_vta_force0_t1_seed0 \
  --configs pinpad --task pinpad_eight --length 20 \
  --out logdir/pinpad_cloud1/pinpad_vta_force0_t1_seed0/boundary_grid.png \
  --internal_out logdir/pinpad_cloud1/pinpad_vta_force0_t1_seed0/boundary_internal.png \
  --fired_out logdir/pinpad_cloud1/pinpad_vta_force0_t1_seed0/boundary_fired.png

python3 scripts/dreamerv3/vta_boundary_eval.py \
  --logdir logdir/pinpad_cloud1/pinpad_vta_force0_t1_seed0 \
  --configs pinpad --task pinpad_eight \
  --episodes 1 --delta_percentile 90 --permutations 50 \
  --out_json logdir/pinpad_cloud1/pinpad_vta_force0_t1_seed0/boundary_eval.json
```

---

## 8. 停止・再開・回収

### 8.1 停止
```bash
pgrep -fl "src_dreamerv3.dreamer|scripts/run_pinpad_sweep.sh" || true
kill <PID...>
```

### 8.2 ログ回収（ローカルへ）
クラウドVM → ローカル（例: rsync）:
```bash
rsync -av <USER>@<HOST>:~/stable-deep-world-model/logdir/pinpad_cloud1 ./logdir/
```

### 8.3 再現性チェック（最低限）
- `git rev-parse HEAD` を記録
- `pip freeze > logdir/pinpad_cloud1/pip_freeze.txt`
- `nvidia-smi > logdir/pinpad_cloud1/nvidia_smi.txt`

---

## 9. よくある問題

### Q. `ModuleNotFoundError: No module named 'pinpad'`
- `pip install -e tasks/pinpad` を実行していない可能性があります。

### Q. `latest.pt` が無くて viz/eval が失敗する
- 学習が途中で落ちています。`logdir/.../metrics.jsonl` と `run.log` を確認してください。

### Q. 数値不安定（NaN/Inf）で落ちる
- 実験側は安定化のため VTA を `--precision 32` で回す前提にしています（`scripts/run_pinpad_sweep.sh` で上書き）。

