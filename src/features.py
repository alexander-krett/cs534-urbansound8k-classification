"""Extract MFCC features from UrbanSound8K audio into a single feature table."""

import pathlib

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = pathlib.Path("data/UrbanSound8K")
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_PATH = pathlib.Path("features/mfcc_features.csv")

SAMPLE_RATE = 22050
N_MFCC = 40


def extract_features(file_path: pathlib.Path) -> np.ndarray:
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    # Collapse the time axis so each clip becomes one fixed-length feature vector.
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH)

    rows = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
        file_path = AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
        try:
            feature_vector = extract_features(file_path)
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        rows.append([row["slice_file_name"], row["fold"], row["classID"], row["class"], *feature_vector])

    n_features = len(rows[0]) - 4
    columns = ["slice_file_name", "fold", "classID", "class"] + [f"mfcc_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(rows, columns=columns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(features_df)} feature rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
