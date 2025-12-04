# Stable Deep World Model (VTA)

VTA (Variable Temporal Abstraction) を使った世界モデルの実験コードです。Bouncing Balls と 3D Maze の2環境での学習・可視化スクリプトを含みます。

## セットアップ
- Python 3.9+ を想定
- 推奨: uv を使う場合  
  ```
  uv venv
  source .venv/bin/activate        # Windows: .venv\Scripts\activate
  uv sync                          # pyproject.toml から依存を解決
  ```
- pip を使う場合  
  ```
  pip install -r requirements.txt
  ```
- GPU を使う場合は CUDA 対応の PyTorch を用意してください。

## プロジェクト構造
```
.
├── requirements.txt        # 主要依存ライブラリ
├── scripts/                # 実行用シェルスクリプト
│   ├── train.sh            # Bouncing Balls 学習
│   └── visualize.sh        # 可視化サンプルコマンド
├── main.py                 # 予備エントリポイント（現状未使用）
├── src/                    # 新実装用のプレースホルダ
│   ├── env/__init__.py     # 仮のサンプルコード
│   ├── models/__init__.py  # 予備
│   ├── train.py            # 予備（未実装）
│   ├── utils/__init__.py   # 予備
│   └── visualize/__init__.py
└── src_vta/                # 現行の VTA 実装パッケージ
    ├── config.py           # 実験設定（環境切替・学習ハイパーパラメータ）
    ├── model2.py           # 代替モデル案
    ├── utils.py            # 前処理・可視化・ログ周り
    ├── models/             # モデル実装
    │   ├── components.py   # Encoder/Decoder など下位ブロック
    │   ├── rssm.py         # 階層RSSM本体
    │   ├── vta.py          # 上位ラッパー
    │   └── __init__.py
    ├── data/               # データ生成・環境
    │   ├── bouncing_balls.py
    │   ├── maze_env.py
    │   ├── generate_npz.py
    │   ├── make_dataset.py
    │   └── __init__.py
    └── scripts/            # 実行用Pythonスクリプト
        ├── train_balls.py  # Bouncing Balls 学習ループ
        ├── train_maze.py   # 3D Maze 学習ループ
        └── visualize.py    # 学習済みモデルの可視化
```

## 使い方
- 学習（Bouncing Balls）  
  `python -m src_vta.scripts.train_balls --exp_name vta_bouncing_balls`  
  または `bash scripts/train.sh`

- 学習（3D Maze）  
  `python -m src_vta.scripts.train_maze`  
  `3d_maze_default/train` / `test` 配下に `.npz` データが必要です。無ければ `python -m src_vta.data.generate_npz` で生成してください。

- 可視化  
  `python -m src_vta.scripts.visualize <ckpt_path> --idx 0 --num_samples 10`  
  `bash scripts/visualize.sh` にはサンプルパスを並べています。

## ログ・成果物
- `src_vta/config.py` の `work_dir` 配下に `exp_name` 単位でログとチェックポイントを保存します。
- TensorBoard ログは `<work_dir>/<exp_name>/logs` に出力されます。

## 補足
- `src/` 配下は将来の再構成用に空のスケルトンを置いています。現行コードはすべて `src_vta/` を見てください。
