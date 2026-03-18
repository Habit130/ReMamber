#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PRETRAIN_DIR="${REPO_ROOT}/pretrain"
HF_HOME_DIR="${REPO_ROOT}/hf_cache"
PRETRAIN_FILE="${PRETRAIN_DIR}/vssm_base_0229_ckpt_epoch_237.pth"
PRETRAIN_URL="https://drive.google.com/uc?id=1O9P6XLuWtUxFa70vwrYCRVedRAFutczV"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required for selective_scan compilation" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"
export HF_HOME="${HF_HOME_DIR}"

mkdir -p "${PRETRAIN_DIR}" "${HF_HOME_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

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

echo "Environment setup complete."
