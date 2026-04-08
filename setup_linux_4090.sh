#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="remamber-linux-4090"
CONDA_BASE="$("$(command -v conda)" info --base)"

cd "$REPO_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"

export CONDA_NO_PLUGINS=true
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU

echo "[1/7] Removing old environment if it exists"
conda deactivate >/dev/null 2>&1 || true
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true

echo "[2/7] Creating fresh environment"
if command -v mamba >/dev/null 2>&1; then
  mamba env create -f environment.linux.4090.yml
elif conda env create --help 2>&1 | grep -q "libmamba"; then
  conda env create --solver libmamba -f environment.linux.4090.yml
else
  conda env create --solver classic -f environment.linux.4090.yml
fi

echo "[3/7] Activating environment"
conda activate "$ENV_NAME"

echo "[4/7] Preparing compiler toolchain for torch cpp_extension"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v g++ >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential
  else
    echo "g++ is required but apt-get is unavailable. Install build-essential manually." >&2
    exit 1
  fi
fi

export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"

echo "[5/7] Verifying CUDA and compiler prerequisites"
command -v nvcc >/dev/null
command -v gcc >/dev/null
command -v g++ >/dev/null
gcc --version
g++ --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[6/7] Installing Python dependencies and selective_scan"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-build-isolation --no-cache-dir ./selective_scan

echo "[7/7] Running import checks"
python -c "import numpy, torch, torchvision, timm, transformers, einops, gdown, triton; import selective_scan_cuda_core, selective_scan_cuda_oflex; print('ready', numpy.__version__, torch.__version__)"

cat <<'EOF'

Environment is ready.

Train:
python main.py --model ReMamber_Mamba --data-set plantseg --data-path ../plantseg --caption-index 3 --batch_size 4 --output_dir outputs/plantseg_mamba

Validate:
python main.py --model ReMamber_Mamba --data-set plantseg --data-path ../plantseg --caption-index 3 --eval --eval-split val --resume outputs/plantseg_mamba/best_checkpoint.pth --output_dir outputs/plantseg_mamba

Test:
python main.py --model ReMamber_Mamba --data-set plantseg --data-path ../plantseg --caption-index 3 --eval --eval-split test --resume outputs/plantseg_mamba/best_checkpoint.pth --output_dir outputs/plantseg_mamba
EOF
