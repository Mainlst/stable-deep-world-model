# トレーニングサマリー (Bounce Balls、RTX 3070)

- 実行: `python -m src_vta.scripts.train_balls --config configs/bounce_balls_3070.json`
- 設定: `configs/bounce_balls_3070.json`
- バッチサイズ: 160、最大反復回数: 50,000、エポックデータサイズ: 24,000 (固定)、use_amp: true
- 作業ディレクトリ: `experiments5_test`、実験名: `vta_bounce_balls_3070`
- 環境: CUDA (AMP with GradScaler)、cuDNNベンチマーク有効、DataLoader pin_memory/num_workers=4/persistent_workers
- ログ: `logs/train.log` (完全なトレースは以下を参照)最後のチェックポイント/可視化は `experiments5_test/` に保存されました。
- 最終観測ステータス（ログ終了）:
- 50,000 ステップ後にトレーニングが終了しました。
- 最後に報告された損失は約 324.45、ベータは約 0.562 です（プログレスバー `L:324.45|B:0.562`）。
- 可視化出力: `experiments5_test/vis_seq_0.png`。

注記:
- BCE の入力/ターゲットは [0,1] にクランプされ、NaN/Inf は CUDA アサートを回避するためにサニタイズされています。
- 今後の作業: AMP API 呼び出しを `torch.amp.autocast('cuda')` / `torch.amp.GradScaler('cuda')` にアップグレードして、非推奨警告を抑制します。