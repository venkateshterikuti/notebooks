#!/usr/bin/env bash
set -euo pipefail

# Single-GPU training launcher for an H100 VM (no Slurm)

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ${SCRATCH_ENV_NAME:-scratch-transformer}

export SCRATCH_DEVICE=cuda

mkdir -p checkpoints

python experiments/run_tinyshakespeare.py \
  --config experiments/tinyshakespeare.yaml \
  --device cuda \
  --out checkpoints/tinyshakespeare-$(date +%Y%m%d-%H%M%S).pkl \
  --save_every 1000

echo "[train] Done. Checkpoints are in ./checkpoints"

