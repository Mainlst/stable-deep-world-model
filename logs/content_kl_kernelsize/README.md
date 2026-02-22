# content_kl_kernelsize quick experiment

## Files
- `run_content_kl_kernel_sweep.sh`: Train `kernel_size={1,3,5}` and run KL/z-distance evaluations.
- `summarize_kernel_results.py`: Aggregate per-kernel metrics into `summary.csv`.

## Default behavior
- Task: `atari_frostbite`
- Config: `atari100k`
- Seed: `0`
- Kernel sizes: `1 3 5`
- Steps: `120000` (quick run)
- GPU: `CUDA_VISIBLE_DEVICES=0`, device=`cuda:0`

## Run
From project root:

```bash
bash logs/content_kl_kernelsize/run_content_kl_kernel_sweep.sh
```

## Useful overrides

```bash
TASK=atari_krull STEPS=200000 GPU_ID=0 bash logs/content_kl_kernelsize/run_content_kl_kernel_sweep.sh
```

```bash
TASK=atari_private_eye KERNEL_SIZES="3 5" SEED=1 COMPILE=True bash logs/content_kl_kernelsize/run_content_kl_kernel_sweep.sh
```

## Outputs
- Runs are saved under:
  - `logs/content_kl_kernelsize/runs/<timestamp>/...`
- Per run:
  - `analysis/vta_abs_ctx_kl_series.csv`
  - `analysis/boundary_stats.csv`
  - `analysis/zt_zt_1.csv`
  - plots (`*.png`)
- Global summary:
  - `logs/content_kl_kernelsize/runs/<timestamp>/summary.csv`

## Note on evaluation constraints
`vta_boundary_viz.py` is executed with:
- `--vta_boundary_force_scale 0`
- `--vta_max_seg_len 1000000`
- `--vta_max_seg_num 1000000`

so boundary forcing constraints are effectively disabled in that evaluation step.
