#!/bin/bash
set -euo pipefail

module load cuda/12.2

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "[error] Conda initialization script not found at ~/miniconda3." >&2
  echo "Install Miniconda and rerun this setup." >&2
  exit 1
fi

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
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup] Downloading TinyShakespeare corpus (idempotent)"
python scripts/download_tinyshakespeare.py

echo "[setup] Verifying installation with a quick model shape test"
python -m tests.test_shapes

echo "[setup] Done. Environment '${ENV_NAME}' is ready."
