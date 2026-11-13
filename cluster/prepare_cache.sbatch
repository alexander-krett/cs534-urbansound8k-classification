#!/bin/bash
#SBATCH --job-name=urbansound-cache
#SBATCH --account=cs534
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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

python src/train_baselines.py \
    --model both \
    --data-dir "${URBANSOUND_DATA_DIR:-data/UrbanSound8K}" \
    --prepare-cache-only
