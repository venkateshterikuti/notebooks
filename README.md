### Prime Intellect H100: simple step-by-step (no Slurm)

These instructions assume you rented a single H100 VM on Prime Intellect (or any Linux GPU VM) and want a quick start without Slurm.

1) Login to the VM and clone the repo

```bash
git clone https://github.com/yourname/transformer-from-scratch.git
cd transformer-from-scratch
```

2) One-time environment setup

```bash
bash scripts/prime_single_setup.sh
```

What this does:
- Installs Miniconda (if missing)
- Creates env `scratch-transformer` with Python 3.11
- Installs `requirements.txt` (includes NumPy and CuPy for CUDA 12)
- Downloads TinyShakespeare
- Runs a quick shape test

3) Start training on GPU

```bash
bash scripts/prime_single_train.sh
```

This launches `experiments/run_tinyshakespeare.py` with `--device cuda` and saves checkpoints in `checkpoints/`.

4) Evaluate perplexity on GPU or CPU

```bash
# GPU
python scripts/eval_ppl.py --checkpoint checkpoints/last.pkl --device cuda

# or CPU
python scripts/eval_ppl.py --checkpoint checkpoints/last.pkl --device cpu
```

5) Sample text from a checkpoint

```bash
python scripts/sample.py --checkpoint checkpoints/last.pkl --device cuda --start "To be, or not to be"
```

Notes
- You do not need Slurm on a single VM. The provided `scripts/prime_train.slurm` is only for HPC clusters that use Slurm.
- If CuPy fails to import, ensure your VM image has NVIDIA drivers and CUDA 12.x. `nvidia-smi` should work.
- To use a different Python version or env name:
  ```bash
  export SCRATCH_PYTHON_VERSION=3.11
  export SCRATCH_ENV_NAME=my-scratch
  bash scripts/prime_single_setup.sh
  ```


