# CS534 UrbanSound8K Classification

Sound event classification on the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt
```

## Dataset

This repo does not include the dataset. Download UrbanSound8K from the
[official site](https://urbansounddataset.weebly.com/urbansound8k.html) (requires
filling out a short form) and unzip it so the layout looks like:

```
data/
  UrbanSound8K/
    audio/
      fold1/ ... fold10/
    metadata/
      UrbanSound8K.csv
```

## Project layout

```
src/
  train_baselines.py # train and evaluate the log-Mel CNN
  compare_svm.py     # reproduce Katie's comparison from her feature CSVs
cluster/         # Turing Slurm submission scripts
data/            # UrbanSound8K goes here (gitignored)
features/        # cached extracted features (gitignored)
models/          # saved CNN predictions and local run artifacts
```

## Usage

```bash
# Cache the CNN spectrograms (one-time)
uv run python src/train_baselines.py --model cnn --data-dir data/UrbanSound8K --prepare-cache-only

# Evaluate the CNN on all ten official test folds
uv run python src/train_baselines.py --model cnn --data-dir data/UrbanSound8K --seed 122333 --epochs 30 --batch-size 64
```

Evaluation follows UrbanSound8K's recommended protocol: 10-fold cross-validation
using the dataset's predefined folds (never re-shuffle/split the data randomly,
since clips from the same source recording are grouped into the same fold).
The CNN uses four convolution blocks and selects its epoch count using the next
official fold cyclically, then refits on all nine non-test folds with that count
(early stopping patience 5). Focused synthetic tests are available with
`uv run python -m unittest discover -s tests -v`.

## Turing (WPI)

Turing uses Slurm. The repository includes `cluster/run_cnn.sbatch` for CNN
training and `cluster/prepare_cache.sbatch` to cache its inputs on CPU.
The jobs expect a `.venv` created in
the repository root. UrbanSound8K is available on scratch at
`/scratch/$USER/urbansound8k/UrbanSound8K/UrbanSound8K` after Kaggle extraction.
On Turing, activate the Python environment from the repository root:

```bash
source /etc/profile
module load python/3.12.10
source .venv/bin/activate
```

Install `requirements.txt` into this environment once if it has not already
been prepared.

Create the log directory and prepare the CNN feature cache once:

```bash
mkdir -p slurm-logs
export URBANSOUND_DATA_DIR="/scratch/$USER/urbansound8k/UrbanSound8K/UrbanSound8K"
sbatch --export=ALL cluster/prepare_cache.sbatch
```

After the cache job finishes successfully, submit the CNN job:

```bash
sbatch --export=ALL cluster/run_cnn.sbatch
squeue --me
```

The jobs request the `cs534` Slurm account and the `academic` partition.
The cache job uses 4 CPUs and 16 GB RAM. The CNN job uses 4 CPUs, one GPU,
16 GB RAM, and a four-hour time limit. It uses seed `122333` and trains up to
30 epochs with early-stopping patience 5. The metrics JSON, predictions CSV,
confusion matrix PNG, and per-fold checkpoints are saved under `models/`.

Turing requires SSH access from the WPI network or GlobalProtect VPN. See the
[Turing getting started guide](https://docs.turing.wpi.edu/getting-started/),
[Slurm guide](https://docs.turing.wpi.edu/getting-started/slurm_guide/), and
[job submission guide](https://docs.turing.wpi.edu/getting-started/submitting/)
for access, account associations, and scheduler use.

## Reproduce the SVM comparison

The report's MFCC, CLAP, and BEATs results come from the common-subset
comparison on Katie's `svm_katie` branch, at commit `ad4f891`. Use the three
feature CSVs produced by that revision:

```bash
uv run python src/compare_svm.py \
  --mfcc /path/to/mfcc_features.csv \
  --clap /path/to/clap_features.csv \
  --beats /path/to/beats_features.csv \
  --output-dir models/svm_comparison
```

The script intersects the three feature tables before cross-validation.
The original intersection has 8,713 clips because the BEATs extractor
skipped 19 very short clips. Outputs include held-out predictions, metrics,
and confusion matrices for the shared clips.
