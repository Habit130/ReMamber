# ReMamber Linux Server Runbook

## Target

- Linux server only
- Single NVIDIA RTX 4090
- Custom binary segmentation dataset at the workspace sibling path `../dataset`
- Training without any `val` logic
- Final evaluation only on `test`

## Required directory layout

Assume the workspace root contains:

```text
<workspace_root>/
  ReMamber/
  dataset/
    train.json
    test.json
    train/
      img/
      lbl/
    test/
      img/
      lbl/
```

The repository now resolves `custom_binary` data to `../dataset` automatically when `--data-path` is left at its default.

## Environment setup

Run the Linux setup helper from the repository root:

```bash
bash setup_server_linux_conda.sh
```

This creates the conda environment `remamber-4090`, installs the locked dependencies, builds `selective_scan`, downloads the official VMamba pretrain checkpoint into `pretrain/`, and prefetches `openai/clip-vit-large-patch14` into the local Hugging Face cache.

The `selective_scan` install is intentionally executed with `--no-build-isolation`, and the environment explicitly pins `numpy<2` plus a `setuptools` version that still provides `pkg_resources`, because its `setup.py` imports `torch` during build time.

## Training

From the repository root:

```bash
conda activate remamber-4090
python main.py \
  --model ReMamber_Mamba \
  --data-path ../dataset \
  --data-set custom_binary \
  --caption-index 2 \
  --output_dir outputs/remamber_mamba_custom
```

Notes:

- `custom_binary` uses `train.json` only during training.
- The training path no longer creates or evaluates a validation split.
- `outputs/remamber_mamba_custom/checkpoint.pth` is the rolling last checkpoint.

## Final evaluation

Use the final checkpoint on the `test` split only:

```bash
conda activate remamber-4090
python main.py \
  --model ReMamber_Mamba \
  --data-path ../dataset \
  --data-set custom_binary \
  --caption-index 2 \
  --resume outputs/remamber_mamba_custom/checkpoint.pth \
  --eval
```

The final test output reports these percentage metrics:

- `IoU`
- `Dice`
- `Recall`
- `mIoU`
- `mAcc`

Metric policy:

- `IoU / Dice / Recall` are foreground-only.
- `mIoU / mAcc` are the mean of foreground and background class metrics.

## Non-Git assets

These are runtime assets and must stay out of Git:

- `../dataset`
- `pretrain/`
- `remamber-4090` conda environment
- `.cache/` or `hf_cache/`
- `outputs/`
- compiled extension artifacts under `selective_scan/`

## Manual prerequisites

- Linux with a working NVIDIA driver and CUDA toolkit with `nvcc`
- Network access for Google Drive and Hugging Face, or equivalent internal mirrors
- Sufficient disk space for the environment, cache, pretrain weights, and training outputs
