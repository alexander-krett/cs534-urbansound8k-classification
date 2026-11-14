#!/bin/bash
#SBATCH --job-name=urbansound-afm
#SBATCH --account=cs534
#SBATCH --partition=academic
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

source /etc/profile
module load python/3.12.10

if [[ ! -d .venv ]]; then
    echo "Missing .venv. See README.md for environment setup." >&2
    exit 1
fi
source .venv/bin/activate

# Keep numerical libraries and PyTorch within the requested CPU allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

# Install and validate dependencies on the allocated compute node.
# Strict shell mode stops the job if either step fails.
python -m pip install -r requirements-afm.txt
python -m pip check

DATA_DIR="${URBANSOUND_DATA_DIR:-data/UrbanSound8K}"
CACHE_DIR="${AFM_CACHE_DIR:-features/afm}"
CHECKPOINT_DIR="${AFM_CHECKPOINT_DIR:-checkpoints/afm}"
OUTPUT_DIR="${AFM_OUTPUT_DIR:-models/afm}"

# Setup fetches model weights and source required by extraction. It runs only
# inside the allocated Slurm job, then the full pipeline reuses those files.
python src/run_afm.py \
    --stage setup \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --baseline-dir models \
    --device cuda \
    --seed 122333 \
    --epochs 100 \
    --batch-size 64

python src/run_afm.py \
    --stage all \
    --data-dir "$DATA_DIR" \
    --cache-dir "$CACHE_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --baseline-dir models \
    --device cuda \
    --seed 122333 \
    --epochs 100 \
    --batch-size 64
