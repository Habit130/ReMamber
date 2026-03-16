# ReMamber Linux Server Workflow

This repository is prepared for a remote Linux server with a single RTX 4090.

## Data layout

The custom dataset is expected to live next to the repository:

```text
workspace/
  ReMamber/
  dataset/
    train.json
    test.json
    train/
      img/
      lbl/
      json/
    test/
      img/
      lbl/
      json/
```

Only `train.json` and `test.json` are used as the dataset index. The per-sample
polygon JSON files remain untouched and are not read by the training pipeline.

## Captions and masks

- Training and testing always use `caption[2]`.
- Masks are binarized on load: `0` stays background and any non-zero value is
  treated as foreground.
- There is no validation split for the custom dataset.

## Environment

Use `environment.server.yml` as the primary Linux environment definition.

The environment includes:

- Python 3.10.13
- PyTorch 2.1.1 with CUDA 11.8
- TorchVision 0.16.1 and Torchaudio 2.1.1
- the repository requirements
- `triton`, `gdown`, and `huggingface_hub`

## Official assets

Use `tools/download_official_weights.py` to prepare non-Git assets:

- required VMamba ImageNet pretrain checkpoint in `pretrain/`
- optional official ReMamber checkpoints in `checkpoints/`
- CLIP text encoder assets in the Hugging Face cache

The repository does not store these assets in Git.

## Training

Use `main.py` as the training entrypoint.

- For the custom dataset, point `--data-path` at `../dataset`.
- Use `--data-set custom_binary`.
- The server-first default model is `ReMamber_Mamba`.
- Custom-dataset training does not run validation or test during epochs.
- The latest training state is written to `checkpoint.pth`.
- The final training state is written to `checkpoint_final.pth`.

## Post-training evaluation

Use `main.py` with `--eval` to run the formal test pass on `test.json`.

The evaluation output is:

- terminal metrics
- `eval_metrics.json` in the chosen output directory

The custom binary test metrics are:

- `iou`
- `dice`
- `recall`
- `miou`
- `macc`

All custom binary metrics are reported as percentages.
