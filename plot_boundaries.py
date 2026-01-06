import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Path to eval_eps
eval_dir = Path('/home/user/stable-deep-world-model/logdir/pinpad_full10/pinpad_vta_force10_t1_seed0/eval_eps')

# Get list of npz files
npz_files = sorted(eval_dir.glob('*.npz'))

# Load the latest one for example
if npz_files:
    latest_npz = npz_files[-1]
    data = np.load(latest_npz)
    print("Keys in npz:", list(data.keys()))
    
    # Assuming 'video' and 'boundaries' or similar
    if 'video' in data:
        video = data['video']  # Shape: (T, H, W, C)
        print("Video shape:", video.shape)
    
    if 'boundaries' in data:
        boundaries = data['boundaries']
        print("Boundaries shape:", boundaries.shape)
        
        # Find steps where boundary is detected (assuming boundaries is boolean or mask)
        boundary_steps = np.where(boundaries)[0]
        print("Boundary steps:", boundary_steps)
        
        # Plot images at boundary steps
        fig, axes = plt.subplots(1, len(boundary_steps), figsize=(15, 5))
        if len(boundary_steps) == 1:
            axes = [axes]
        for i, step in enumerate(boundary_steps):
            img = video[step]
            axes[i].imshow(img)
            axes[i].set_title(f'Step {step}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig('/home/user/stable-deep-world-model/boundary_detection_plot.png')
        plt.show()
    else:
        print("No boundaries key found")
else:
    print("No npz files found")