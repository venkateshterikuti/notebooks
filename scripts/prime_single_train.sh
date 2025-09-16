#!/usr/bin/env bash
set -euo pipefail

# Single-GPU training launcher for an H100 VM (no Slurm)

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ${SCRATCH_ENV_NAME:-scratch-transformer}

export SCRATCH_DEVICE=cuda

mkdir -p checkpoints artifacts

timestamp=$(date +%Y%m%d-%H%M%S)
run_name="tinyshakespeare-${timestamp}"
checkpoint_path="checkpoints/${run_name}.pkl"
artifact_dir="artifacts/${run_name}"
metrics_csv="${artifact_dir}/training_metrics.csv"
metrics_json="${artifact_dir}/training_metrics.json"
loss_plot="${artifact_dir}/training_loss.png"
log_file="${artifact_dir}/train.log"

mkdir -p "${artifact_dir}"

timestamp=$(date +%Y%m%d-%H%M%S)
run_name="tinyshakespeare-${timestamp}"
checkpoint_path="checkpoints/${run_name}.pkl"
artifact_dir="artifacts/${run_name}"
metrics_csv="${artifact_dir}/training_metrics.csv"
metrics_json="${artifact_dir}/training_metrics.json"
loss_plot="${artifact_dir}/training_loss.png"
log_file="${artifact_dir}/train.log"

mkdir -p "${artifact_dir}"

echo "[train] Launching experiments/run_tinyshakespeare.py (writing log to ${log_file})"

PYTHONUNBUFFERED=1 python -u experiments/run_tinyshakespeare.py \
  --config experiments/tinyshakespeare.yaml \
  --device cuda \
  --out "${checkpoint_path}" \
  --save_every 1000 \
  --metrics_csv "${metrics_csv}" \
  --metrics_json "${metrics_json}" \
  --loss_plot "${loss_plot}" \
  2>&1 | tee "${log_file}"

archive_path="${artifact_dir}.tar.gz"

python scripts/package_artifacts.py \
  --artifact-dir "${artifact_dir}" \
  --checkpoint "${checkpoint_path}" \
  --include "${metrics_csv}" \
  --include "${metrics_json}" \
  --include "${loss_plot}" \
  --include "${log_file}" \
  --archive "${archive_path}" \
  ${SCRATCH_AUTO_DOWNLOAD_TARGET:+--download "${SCRATCH_AUTO_DOWNLOAD_TARGET}"} \
  ${SCRATCH_AUTO_DOWNLOAD_METHOD:+--download-method "${SCRATCH_AUTO_DOWNLOAD_METHOD}"}

echo "[train] Done. Checkpoints are in ./checkpoints"
echo "[artifacts] Run outputs stored in ${artifact_dir}"
if [[ -n "${SCRATCH_AUTO_DOWNLOAD_TARGET:-}" ]]; then
  echo "[artifacts] Archive sync attempted to ${SCRATCH_AUTO_DOWNLOAD_TARGET} using ${SCRATCH_AUTO_DOWNLOAD_METHOD:-rsync}"
fi

