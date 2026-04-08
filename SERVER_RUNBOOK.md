# ReMamber Linux 4090 Runbook

本仓库的服务器交付面固定为单卡 RTX 4090、CUDA 11.8、Python 3.10、`../plantseg` 数据集和 `ReMamber_Mamba`。

## 1. 一键重建环境

```bash
bash ./setup_linux_4090.sh
```

这个脚本会自动：

- 删除旧的 `remamber-linux-4090` 环境
- 重新创建环境
- 处理 MKL 激活变量
- 配置 Conda 编译器到 `gcc/g++`
- 安装 `requirements.txt`
- 编译并安装 `./selective_scan`
- 执行关键导入检查

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
