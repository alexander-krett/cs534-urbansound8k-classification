# CS534 UrbanSound8K Classification

Sound event classification on the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
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
  features.py    # extract MFCC features from audio into a single table
  train_svm.py   # train + evaluate an SVM classifier on extracted features
  train_baselines.py # run the CNN or SVM baseline
cluster/         # Turing Slurm submission scripts
data/            # UrbanSound8K goes here (gitignored)
features/        # cached extracted features (gitignored)
models/          # saved trained models (gitignored)
```

## Usage

```bash
# Extract/cache both baseline feature sets (one-time)
python src/train_baselines.py --model both --data-dir data/UrbanSound8K --prepare-cache-only

# Evaluate all ten official test folds with either model
python src/train_baselines.py --model svm --data-dir data/UrbanSound8K --seed 122333
python src/train_baselines.py --model cnn --data-dir data/UrbanSound8K --seed 122333 --epochs 30 --batch-size 64
```

Evaluation follows UrbanSound8K's recommended protocol: 10-fold cross-validation
using the dataset's predefined folds (never re-shuffle/split the data randomly,
since clips from the same source recording are grouped into the same fold).
The CNN uses four convolution blocks and selects its epoch count using the next
official fold cyclically, then refits on all nine non-test folds with that count
(early stopping patience 5). Focused synthetic tests are available with
`python -m unittest discover -s tests -v`.

## Turing (WPI)

Turing uses Slurm. The repository includes jobs for the CNN and SVM in
`cluster/run_cnn.sbatch` and `cluster/run_svm.sbatch`, plus a CPU feature-cache
job at `cluster/prepare_cache.sbatch`. The jobs expect a `.venv` created in
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

Create the log directory and prepare both feature caches once:

```bash
mkdir -p slurm-logs
export URBANSOUND_DATA_DIR="/scratch/$USER/urbansound8k/UrbanSound8K/UrbanSound8K"
sbatch --export=ALL cluster/prepare_cache.sbatch
```

After the cache job finishes successfully, submit each baseline:

```bash
sbatch --export=ALL cluster/run_svm.sbatch
sbatch --export=ALL cluster/run_cnn.sbatch
squeue --me
```

The jobs explicitly request the `cs534` Slurm account for Prof. Korkin's course
and the `short` partition. The cache job uses 4 CPUs and 16 GB RAM. SVM uses
4 CPUs, 8 GB RAM, and 4 hours. CNN uses 4 CPUs, one GPU, 16 GB RAM, and 4
hours. Both evaluations use random seed `122333`; CNN trains up to 30 epochs
with early-stopping patience 5. Compact JSON reports and run manifests are
saved under `models/`; checkpoints are saved there for each test fold.

Turing requires SSH access from the WPI network or GlobalProtect VPN. See the
[Turing getting started guide](https://docs.turing.wpi.edu/getting-started/),
[Slurm guide](https://docs.turing.wpi.edu/getting-started/slurm_guide/), and
[job submission guide](https://docs.turing.wpi.edu/getting-started/submitting/)
for access, account associations, and scheduler use.
