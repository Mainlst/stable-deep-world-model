import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess(images, bits=None):
    """
    画像を前処理する関数
    bits が指定されている場合、そのビット深度に量子化する
    """
    if bits is None:
        return images
    
    # ビット深度の削減 (例: 5bit = 32段階)
    bins = 2 ** bits
    if hasattr(images, 'device'):
        # Tensorの場合
        images = torch.floor(images * bins) / bins
    else:
        # Numpyの場合
        images = np.floor(images * bins) / bins
    return images

def postprocess(images, bits=None):
    """
    モデルの出力（0-1）を可視化用に調整する関数
    """
    if bits is None:
        return images
    
    # 量子化した場合は、階調の中央値になるように0.5ビン分ずらすと綺麗に見えることがあるが
    # ここでは単純にそのまま返す（あるいは必要なら255倍してintにするなど）
    return images

def visualize_results(model, loader, config, seq_idx=0, return_fig=False):
    model.eval()
    
    # バッチから画像と「真の境界」を取り出す
    batch = next(iter(loader))
    obs = batch[0].to(config.device)      # 画像
    true_hits = batch[1][seq_idx].cpu()   # 真の境界 (Seq,)
    
    # 前処理の適用
    obs = preprocess(obs, config.obs_bit)
    act = torch.zeros(obs.size(0), obs.size(1), config.action_size).to(config.device)

    with torch.no_grad():
        results = model(
            obs, act, config.seq_size, config.init_size, 
            obs_std=config.obs_std, loss_type=config.loss_type
        )

    input_seq = obs[seq_idx].cpu()
    rec_seq = results['rec_data'][seq_idx].cpu()
    
    # Boundary Probs (予測された境界)
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
        # 1行目: Original (真の境界を表示)
        # ------------------------------------------------
        ax = axes[0, t]
        ax.imshow(np.clip(input_seq[config.init_size + t].permute(1, 2, 0).numpy(), 0, 1))
        ax.axis('off')
        
        # 真の境界 (true_hits) があれば「赤色」の枠を表示
        # (init_size分ずらして参照することに注意)
        is_hit = true_hits[config.init_size + t] > 0.5
        
        if is_hit:
            # 正解は赤枠で見やすく
            rect = patches.Rectangle((0, 0), 31, 31, linewidth=9, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
        # ------------------------------------------------
        # 2行目: Rec (予測された境界を表示)
        # ------------------------------------------------
        ax = axes[1, t]
        ax.imshow(np.clip(rec_seq[t].permute(1, 2, 0).numpy(), 0, 1))
        ax.axis('off')
        
        # ★追加: 再構成画像の方に、モデルが予測した「赤色」の枠を表示
        if boundary_probs[t] > 0.5:
            rect = patches.Rectangle((0, 0), 31, 31, linewidth=9, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        # ------------------------------------------------
        # 3行目: Prob (予測確率)
        # ------------------------------------------------
        ax = axes[2, t]
        ax.bar([0], [boundary_probs[t]], color='red' if boundary_probs[t] > 0.5 else 'blue')
        ax.set_ylim(0, 1.1)
        ax.axis('off')
    
    if return_fig:
        return fig
    
    plt.savefig(config.work_dir / f"vis_seq_{seq_idx}.png")
    plt.close()
    print(f"Saved visualization to {config.work_dir}")