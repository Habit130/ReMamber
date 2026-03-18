#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="${REPO_ROOT}/pretrain"
HF_HOME_DIR="${REPO_ROOT}/hf_cache"
PRETRAIN_FILE="${PRETRAIN_DIR}/vssm_base_0229_ckpt_epoch_237.pth"
PRETRAIN_URL="https://drive.google.com/uc?id=1O9P6XLuWtUxFa70vwrYCRVedRAFutczV"
ENV_NAME="${CONDA_ENV_NAME:-remamber-4090}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required for selective_scan compilation" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"
export HF_HOME="${HF_HOME_DIR}"

mkdir -p "${PRETRAIN_DIR}" "${HF_HOME_DIR}"

eval "$(conda shell.bash hook)"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda env create -f "${REPO_ROOT}/environment.server.yml" -n "${ENV_NAME}"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.1 torchvision==0.16.1
python -m pip install -r "${REPO_ROOT}/requirements.txt"
python -m pip install -r "${REPO_ROOT}/requirements-server.txt"
python -m pip install -e "${REPO_ROOT}/selective_scan"

if [ ! -f "${PRETRAIN_FILE}" ]; then
  python -m gdown "${PRETRAIN_URL}" -O "${PRETRAIN_FILE}"
fi

python - <<'PY'
from transformers import CLIPTextModel, CLIPTokenizerFast

model_id = "openai/clip-vit-large-patch14"
CLIPTokenizerFast.from_pretrained(model_id)
CLIPTextModel.from_pretrained(model_id)
print(f"prefetched {model_id}")
PY

echo "Conda environment setup complete: ${ENV_NAME}"
