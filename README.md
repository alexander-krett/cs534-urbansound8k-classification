# CS534 UrbanSound8K Classification

Sound event classification on the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
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
data/            # UrbanSound8K goes here (gitignored)
features/        # cached extracted features (gitignored)
models/          # saved trained models (gitignored)
```

## Usage

```bash
# 1. Extract features (one-time, can take a while)
python src/features.py

# 2. Train and evaluate the SVM
python src/train_svm.py
```

Evaluation follows UrbanSound8K's recommended protocol: 10-fold cross-validation
using the dataset's predefined folds (never re-shuffle/split the data randomly,
since clips from the same source recording are grouped into the same fold).
