'''
便利関数まとめ
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc  # メモリ管理用

def preprocess(images, bits=None):
    if bits is None:
        return images
    bins = 2 ** bits
    if hasattr(images, 'device'):
        images = torch.floor(images * bins) / bins
    else:
        images = np.floor(images * bins) / bins
    return images

def preprocess_maze(image, bits=5):
    """ 画像をモデルへの入力形式に前処理する．

    Parameters
    ----------
        image (torch.Tensor): [0, 1] の画像
        bits (int): 変換後のビット深度
    Returns
    ----------
        torch.Tensor: [-1, 1] に正規化された画像
    """
    bins = 2 ** bits
    # 1. ピクセル値を [0, 255] に戻す
    image = image * 255.0
    # 2. 色のビット深度を減らし量子化
    if bits < 8:
        image = torch.floor(image / 2 ** (8 - bits))
    image = image / bins
    # 3. Dequantization: 離散的なピクセル値にノイズを加えて連続値にする
    image = image + image.new_empty(image.shape).uniform_() / bins
    # 4. [0.0, 1.0] -> [-1.0, 1.0]
    return (image - 0.5) * 2.0

def postprocess(image, bits=None):
    """ モデル出力を可視化するための後処理をする．

    Parameters
    ----------
        image (torch.Tensor): [-1, 1] のモデル出力画像
        bits (int): 前処理で使ったビット深度
    Returns
    ----------
        torch.Tensor: [0, 1] の表示用の画像
    """
    bins = 2 ** bits
    # [-1, 1] -> [0, 1]
    image = image / 2.0 + 0.5
    # [0, 1] -> [0, 2^bits - 1]
    image = torch.floor(bins * image)
    # [0, 2^bits - 1] -> [0, 255]
    image = image * (255.0 / (bins - 1))
    # [0, 255] の範囲にクリッピングし，[0, 1]に正規化
    return torch.clamp(image, min=0.0, max=255.0) / 255.0

def log(partition, results, writer, b_idx):
    """ Tensorboardに書き込み用関数 """
    assert partition in ["train", "test"]

    # compute total loss (mean over steps and seqs)
    obs_cost = results['obs_cost'].mean()
    kl_abs_cost = results['kl_abs_state'].mean()
    kl_obs_cost = results['kl_obs_state'].mean()
    kl_mask_cost = results['kl_mask'].mean()

    # 各曲線に最新の値を追加
    writer.add_scalar(f'{partition}/full_cost', obs_cost + kl_abs_cost + kl_obs_cost + kl_mask_cost, global_step=b_idx)
    writer.add_scalar(f'{partition}/obs_cost', obs_cost, global_step=b_idx)
    writer.add_scalar(f'{partition}/kl_full_cost', kl_abs_cost + kl_obs_cost + kl_mask_cost, global_step=b_idx)
    writer.add_scalar(f'{partition}/kl_abs_cost', kl_abs_cost, global_step=b_idx)
    writer.add_scalar(f'{partition}/kl_obs_cost', kl_obs_cost, global_step=b_idx)
    writer.add_scalar(f'{partition}/kl_mask_cost', kl_mask_cost, global_step=b_idx)
    writer.add_scalar(f'{partition}/read_ratio', results['mask_data'].sum(1).mean(), global_step=b_idx)
    if partition == "train":
        writer.add_scalar(f'{partition}/q_ent', results['p_ent'].mean(), global_step=b_idx)
        writer.add_scalar(f'{partition}/p_ent', results['q_ent'].mean(), global_step=b_idx)
        writer.add_scalar(f'{partition}/beta', results['beta'], global_step=b_idx)

    log_str = f'[%08d] {partition}=elbo:%7.3f, obs_nll:%7.3f, ' \
              'kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, ' \
              'num_reads:%3.1f, beta: %3.3f, ' \
              'p_ent: %3.2f, q_ent: %3.2f'
    log_data = [
        b_idx,
        - (obs_cost + kl_abs_cost + kl_obs_cost + kl_mask_cost),
        obs_cost,
        kl_abs_cost + kl_obs_cost + kl_mask_cost,
        kl_abs_cost,
        kl_obs_cost,
        kl_mask_cost,
        results['mask_data'].sum(1).mean(),
    ]
    if partition == "train":
        log_data += [
            results['beta'],
            results['p_ent'].mean(),
            results['q_ent'].mean(),
        ]
    return log_str, log_data

def visualize_results(model, loader, config, seq_idx=0, return_fig=False):
    model.eval()
    
    # バッチを取得
    batch = next(iter(loader))
    
    # データセットの構造に応じて obs / act / hits を振り分ける
    obs = batch[0].to(config.device)
    true_hits = torch.zeros(obs.size(1))

    if len(batch) > 1:
        candidate = batch[1]
        # 行動が (B, T, A) で提供されている場合のみ act として使う
        if candidate.dim() == 3 and candidate.size(-1) == config.action_size and config.action_size > 0:
            act = candidate.to(config.device)
        else:
            # Bouncing Balls のように (B, T) の衝突ラベルが2番目に来る場合はこちら
            true_hits = candidate[seq_idx].cpu()
            act = torch.zeros(obs.size(0), obs.size(1), config.action_size).to(config.device)
    else:
        act = torch.zeros(obs.size(0), obs.size(1), config.action_size).to(config.device)

    # 真の境界データが第3要素以降にある場合にも対応
    '''
    if len(batch) > 2:
        true_hits = batch[2][seq_idx].cpu()
    '''

    # 前処理
    obs = preprocess(obs, config.obs_bit)

    # 推論実行
    with torch.no_grad():
        results = model(
            obs, act, config.seq_size, config.init_size, 
            obs_std=config.obs_std, loss_type=config.loss_type
        )

    input_seq = obs[seq_idx].cpu()
    rec_seq = results['rec_data'][seq_idx].cpu()
    
    # 予測された境界確率
    if 'q_mask' in results:
        boundary_probs_raw = results['q_mask'][seq_idx].cpu()
        if boundary_probs_raw.shape[-1] == 2:
            boundary_probs = boundary_probs_raw[:, 1]
        else:
            boundary_probs = boundary_probs_raw.squeeze()
    else:
        boundary_probs = torch.zeros(input_seq.shape[0])

    display_len = min(config.seq_size, 20)
    
    fig, axes = plt.subplots(3, display_len, figsize=(2 * display_len, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for t in range(display_len):
        # ------------------------------------------------
        # 1行目: Original (真の画像)
        # ------------------------------------------------
        ax = axes[0, t]
        # 配列の次元順序を (C, H, W) -> (H, W, C) に変更して表示
        ax.imshow(np.clip(input_seq[config.init_size + t].permute(1, 2, 0).numpy(), 0, 1))
        ax.axis('off')
        
        # 真の境界があれば赤枠 (Maze環境ではデータセットに含まれない場合があるので注意)
        is_hit = true_hits[config.init_size + t] > 0.5
        if is_hit:
            rect = patches.Rectangle((0, 0), 31, 31, linewidth=4, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
        # ------------------------------------------------
        # 2行目: Rec (再構成画像 + 予測境界)
        # ------------------------------------------------
        ax = axes[1, t]
        ax.imshow(np.clip(rec_seq[t].permute(1, 2, 0).numpy(), 0, 1))
        ax.axis('off')
        
        # モデルが予測した境界を赤枠で表示
        if boundary_probs[t] > 0.5:
            # linewidthを少し調整
            rect = patches.Rectangle((0, 0), 31, 31, linewidth=4, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        # ------------------------------------------------
        # 3行目: Prob (境界確率のグラフ)
        # ------------------------------------------------
        ax = axes[2, t]
        ax.bar([0], [boundary_probs[t]], color='red' if boundary_probs[t] > 0.5 else 'blue')
        ax.set_ylim(0, 1.1)
        ax.axis('off')
        # タイトルに行動を表示するとわかりやすい（オプション）
        # action_idx = torch.argmax(act[seq_idx, config.init_size + t]).item()
        # acts_str = ["FWD", "LFT", "RGT"]
        # if action_idx < 3: ax.set_title(acts_str[action_idx], fontsize=8)

    if return_fig:
        return fig
    
    save_path = config.work_dir / f"vis_seq_{seq_idx}.png"
    plt.savefig(save_path)
    # ★修正箇所: 単なる plt.close() ではなく、以下のように変更して確実にメモリを解放
    plt.close('all') 
    plt.clf()
    gc.collect() # Pythonのガベージコレクションを強制実行
    print(f"Saved visualization to {save_path}")
