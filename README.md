# Stable Deep World Model (VTA)

VTA (Variable Temporal Abstraction) を使った世界モデルの実験コードです。Bouncing Balls と 3D Maze の2環境での学習・可視化スクリプトを含みます。

## 研究進捗
1. 関連研究について複数論文を調査できている．
    
   ---
    
    - VTA（ベースライン）
    - THICK（主な比較対象）
    - Hieros（固定ステップ数での階層化手法1）
    - CW-VAE（固定ステップ数での階層化手法2）
    - Director（固定ステップ数での階層化手法3）
    
    |  | **階層数** | **抽象化幅（固定 v.s. 動的）** | **方策学習** | **オンライン** |
    | --- | --- | --- | --- | --- |
    | **VTA** | 2 | 動的 | × | × |
    | **LOVE** | 2 | 動的 | 〇 | × |
    | **THICK** | 2 | 動的 | 〇 | 〇 |
    | **Hieros** | 2 ~ 3 | 固定 | 〇 | 〇 |
    | **CW-VAE** | 2 ~ 3 | 固定 | × | × |
    | **Director** | 2 | 固定 | 〇 | 〇 |
    | **Ours** | 2 ~ | 動的 | 〇 | 〇 |
   
2. ベースラインとなる研究を選定できている．
    
    ---
    
    固定ステップ数での時間抽象化を行わず，かつ時間階層を3層以上に増やすことが可能なVTAを階層化手法のベースラインとして利用．
    
    世界モデルのベースとしては汎化性能に優れたDreamerV3を使用．
   
3. ベースラインモデル（もしくはその再現実装）を動かせている．
    
    ---
    
    VTAの概念を階層化の手法として利用し，DreamerV3のrssmを置き換える．
    
    現状は，単純に抽象状態zの層を追加し，観測状態と結合して方策ネットワークに入力する形で利用．
    
    結果としてDreamerV3と同程度の性能は出せたものの，方策学習にはあまり寄与していないことが実験から分かった．
   
4. 仮説を立てながら提案手法の実験を進められている．
    
    ---
    
    ベースライン（VTA）が行う境界検出にはいくつかの課題がある．
    
    1. VTAは観測が大きく切り替わる瞬間しか捉えることができないため，Atariのようなタスクでは意味のある区切りを発見することができず，ステージ遷移のようなわかりやすい区切りの発見にとどまっているため，方策学習にはあまり寄与できない．
    2. 単純に抽象状態zを観測状態sと結合して方策選択するだけでは境界を発見した意味が長期的な文脈の保持に留まってしまい，DreamerV3の性能を超えることは難しい．
    
    これに対して以下の仮説を立てる．
    
    1. DreamerV3は内部で状態価値を推定しているため，この情報を境界検出に組み込むことでタスクに有意義な境界を発見することにつながる．
    2. Directorのように，Goal AEを学習することで抽象状態から現在何をするべきかを抽出し，方策学習に役立てられる．
    
    1の仮説では，状態価値は状態がどの程度報酬に近づいているかを表現しているため，その情報は鍵を入手した，敵を倒したといった観測だけでは得にくい情報も保持していることを意味していると考えているためである．
    
    2の仮説では，境界として検出した情報を長期的な文脈の保持だけでなく，最大限有意義に用いるためには，抽象化したゴール（サブゴール）を観測レベルに指示することが効果的だと考えているためである．
   
## DreamerV3 (torch) 統合
DreamerV3 の PyTorch 実装を `src_dreamerv3/` として統合しています。詳細は `docs/dreamerv3/README.md` を参照してください。

- 依存インストール: `pip install -r requirements-dreamerv3.txt`
- 例: `python -m src_dreamerv3.dreamer --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk`

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
├── configs/                # 設定ファイル（JSON）
│   └── bouncing_balls_3070.json
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
- 設定ファイル（JSON）  
  `configs/bouncing_balls_3070.json`（Bouncing Balls）と `configs/3d_maze_default.json`（3D Maze）のサンプルを用意しています。自分の環境に合わせて値を調整してください。

- 学習（Bouncing Balls）  
  `python -m src_vta.scripts.train_balls --config configs/bouncing_balls_3070.json`  
  または `bash scripts/train.sh`

- 学習（3D Maze）  
  `python -m src_vta.scripts.train_maze --config <your_maze_config.json>`  
  デフォルトで `data/3d_maze_default/{train,test}` 配下の `.npz` を参照します（`Config.maze_data_dir` で変更可能）。データが無ければ `python -m src_vta.data.generate_npz --out <保存先>` で生成してください。

- 可視化  
  `python -m src_vta.scripts.visualize <ckpt_path> --config configs/bouncing_balls_3070.json --idx 0 --num_samples 10`  
  `bash scripts/visualize.sh` にはサンプルパスを並べています。

## 設定ファイルについて
- フォーマットは JSON。`configs/bouncing_balls_3070.json` をベースに環境に合わせて編集してください。
- 未知のキーは無視され、`Config` クラスに存在するキーのみ上書きされます。
- `*_dir` キーはパスとして扱われ自動作成されます。3D Maze 用のデータセット場所は `maze_data_dir` で指定できます。

## ログ・成果物
- `src_vta/config.py` の `work_dir` 配下に `exp_name` 単位でログとチェックポイントを保存します。
- TensorBoard ログは `<work_dir>/<exp_name>/logs` に出力されます。
- 学習時に Weights & Biases へも自動でメトリクスを送信します（`project`: `stable-deep-world-model`）。環境変数 `WANDB_MODE=offline` でオフライン実行も可能です。

## 補足
- `src/` 配下は将来の再構成用に空のスケルトンを置いています。現行コードはすべて `src_vta/` を見てください。
