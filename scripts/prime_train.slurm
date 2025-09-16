#!/bin/bash
#SBATCH -J scratch_transformer
#SBATCH --partition=pi_h100            # adjust to the exact Prime Intellect partition name
#SBATCH --account=REPLACE_WITH_ACCOUNT  # update to your allocation
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH -o /groups/atila/wan22/logs/%x-%j.out

module load cuda/12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scratch-transformer

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PROJECT_ROOT="/groups/atila/wan22/transformer-from-scratch"
cd "$PROJECT_ROOT"

if [ ! -f data/tinyshakespeare.txt ]; then
  python scripts/download_tinyshakespeare.py
fi

python experiments/run_tinyshakespeare.py \
  --config experiments/tinyshakespeare.yaml \
  --device cuda \
  --out checkpoints/tinyshakespeare-${SLURM_JOB_ID}.pkl \
  --save_every 1000
