# ReMamber Linux 4090 Runbook

本仓库的服务器交付面固定为单卡 RTX 4090、CUDA 11.8、Python 3.10、`../plantseg` 数据集和 `ReMamber_Mamba`。

## 1. 环境准备

```bash
conda env create -f environment.linux.4090.yml
conda activate remamber-linux-4090
python -m pip install --no-build-isolation ./selective_scan
```

如果环境已创建但包状态异常，执行下面这组修复命令：

```bash
conda activate remamber-linux-4090
python -m pip install --upgrade pip
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./selective_scan
```

建议在训练前先做一次导入检查：

```bash
python -c "import numpy, torch, torchvision, timm, transformers, einops, gdown, triton; print('numpy', numpy.__version__)"
```

## 2. 训练

```bash
python main.py --model ReMamber_Mamba --data-set plantseg --data-path ../plantseg --caption-index 3 --batch_size 4 --output_dir outputs/plantseg_mamba
```

说明：
- 首次运行会自动下载 VMamba backbone 预训练权重到 `pretrain/`
- 首次运行会自动下载 CLIP tokenizer 和 text encoder 到 `pretrain/huggingface/`
- 若 4090 显存不足，可把 `--batch_size 4` 改为 `--batch_size 2`

## 3. Val 集评估

```bash
python main.py --model ReMamber_Mamba --data-set plantseg --data-path ../plantseg --caption-index 3 --eval --eval-split val --resume outputs/plantseg_mamba/best_checkpoint.pth --output_dir outputs/plantseg_mamba
```

评估结果会写到 `outputs/plantseg_mamba/eval_val_metrics.json`。

## 4. Test 集最终评估

```bash
python main.py --model ReMamber_Mamba --data-set plantseg --data-path ../plantseg --caption-index 3 --eval --eval-split test --resume outputs/plantseg_mamba/best_checkpoint.pth --output_dir outputs/plantseg_mamba
```

评估结果会写到 `outputs/plantseg_mamba/eval_test_metrics.json`，指标固定为：
- `IoU`
- `Dice`
- `Recall`
- `mIoU`
- `mACC`
