#!/usr/bin/env bash
set -euo pipefail

# Simple one-time setup for a single-GPU H100 VM (no Slurm required)

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found. Use a GPU image or install NVIDIA drivers (CUDA 12+)." >&2
  exit 1
fi

echo "[info] GPU detected:"
nvidia-smi || true

# Install Miniconda if missing
if [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "[setup] Installing Miniconda (user)"
  curl -fsSL -o "$HOME/miniconda.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

ENV_NAME=${SCRATCH_ENV_NAME:-scratch-transformer}
PYTHON_VERSION=${SCRATCH_PYTHON_VERSION:-3.11}

if ! conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "[setup] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
else
  echo "[setup] Using existing conda environment '${ENV_NAME}'"
fi

conda activate "$ENV_NAME"

echo "[setup] Upgrading pip and installing requirements"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[setup] Verifying CuPy can see your GPU"
python - <<'PY'
import os
os.environ.setdefault("SCRATCH_DEVICE", "cuda")
import cupy as cp
name = cp.cuda.runtime.getDeviceProperties(0)['name']
print("[ok] CUDA device:", name)
PY

echo "[setup] Downloading TinyShakespeare corpus (idempotent)"
python scripts/download_tinyshakespeare.py

echo "[setup] Quick model shape test"
python -m tests.test_shapes

echo "[setup] Done. Activate with:  source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"

