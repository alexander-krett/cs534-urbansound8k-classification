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
and the `academic` partition. The cache job uses 4 CPUs and 16 GB RAM. SVM uses
4 CPUs, 8 GB RAM, and 4 hours. CNN uses 4 CPUs, one GPU, 16 GB RAM, and 4
hours. Both evaluations use random seed `122333`; CNN trains up to 30 epochs
with early-stopping patience 5. Compact JSON reports and run manifests are
saved under `models/`; checkpoints are saved there for each test fold.

### Audio foundation model (AFM)

The AFM pipeline uses ten fixed UrbanSound8K class prompts of the form
`An audio recording of ...`. It extracts frozen LAION CLAP
`630k-audioset-best` audio embeddings (fusion disabled, 512 dimensions) and
self-supervised BEATs iter3+AS2M embeddings (mean pooled, 768 dimensions), then
L2-normalizes and concatenates them into a 1280-dimensional fused
representation. It runs seven evaluation branches over the ten official
folds: CLAP zero-shot classification, plus SVM and MLP probes on each of the
CLAP, BEATs, and fused representations. Prompts are fixed; there is no prompt
search or ensembling.

SVM probes use fixed `C=10` and `gamma=scale`. For each MLP test fold, the
epoch count is selected on the next official fold cyclically; a fresh scaler
and model are then fit on all nine non-test folds for that selected number of
epochs. Metrics include accuracy and macro and weighted precision, recall,
and F1, reported as per-fold mean and standard deviation and as pooled
predictions. Baseline SVM/CNN comparisons are included only when their saved
manifests are compatible with the AFM provenance; otherwise they are marked
unavailable.

The AFM batch job installs `requirements-afm.txt` into the existing `.venv`
and runs `python -m pip check` before checkpoint setup and evaluation. Both
steps execute on the allocated compute node and stop the job on failure;
no separate dependency installation command is needed. Create `.venv` using
the project setup instructions if it does not already exist.

`requirements-afm.txt` includes the base requirements and `laion-clap==1.1.7`,
with pinned NumPy/Pandas/Transformers and PyTorch/audio/vision versions for
Python 3.12. Installation may change versions in the existing `.venv`.
Dependency installation shares the job's GPU allocation and four-hour limit.

From the repository root, submit the AFM job as follows. The log directory
must exist before submission. The job requests one GPU, 4 CPUs, 32 GB RAM, and
4 hours on the `academic` partition under account `cs534`; failures stop the
pipeline immediately.

```bash
mkdir -p slurm-logs
export URBANSOUND_DATA_DIR="/scratch/$USER/urbansound8k/UrbanSound8K/UrbanSound8K"
sbatch --export=ALL cluster/run_afm.sbatch
squeue --me
```

After dependency installation and validation, the batch file runs `setup`
and then `all` inside the allocated GPU job, so
model weights and source are fetched on Turing rather than on the login node.
Setup downloads the CLAP checkpoint and pinned BEATs source. BEATs weights are
hosted on Microsoft OneDrive; if the built-in direct download link is
unavailable, obtain the official iter3+ AS2M checkpoint and provide either a
direct download URL via `BEATS_CHECKPOINT_URL` or put the file at
`checkpoints/afm/BEATs_iter3_plus_AS2M.pt` before running extraction.
By default, extracted embeddings are cached in `features/afm/`, model/source
files are placed in `checkpoints/afm/`, and AFM reports, manifests, and
per-fold checkpoints are written to `models/afm/`. The output includes CSV
prediction tables, PNG confusion matrices, JSON metrics/manifests, and
LaTeX/Markdown summary tables.

For individual stages, invoke `python src/run_afm.py --stage setup|extract|evaluate|all`
with the same path and device options from within a Slurm GPU allocation.
`setup` downloads the encoder assets, `extract` builds or reuses the embedding
cache, and `evaluate` requires an existing valid cache and runs all seven evaluation
branches across ten folds. A real-model encoder
smoke run and CUDA-backed extraction require a Slurm GPU allocation;
`--stage smoke` is intended for that allocation. For a lightweight local CLI
check without downloading model weights, use
`.venv/bin/python src/run_afm.py --synthetic-smoke`.

Run lightweight synthetic and mocked regression checks with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v`.
Real-checkpoint integration and full accuracy require a Slurm run and are not
verified by these checks. The entry point checks that its actual host belongs
to the allocation; after `salloc`, use `srun` to execute on a compute node.

Turing requires SSH access from the WPI network or GlobalProtect VPN. See the
[Turing getting started guide](https://docs.turing.wpi.edu/getting-started/),
[Slurm guide](https://docs.turing.wpi.edu/getting-started/slurm_guide/), and
[job submission guide](https://docs.turing.wpi.edu/getting-started/submitting/)
for access, account associations, and scheduler use.
