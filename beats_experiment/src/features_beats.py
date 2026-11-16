"""Extract BEATs embeddings from UrbanSound8K audio into a single feature table."""

import argparse
import pathlib
import sys

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent / "beats"))
from BEATs import BEATs, BEATsConfig  # noqa: E402

EXPERIMENT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = pathlib.Path("data/UrbanSound8K")
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATA_DIR / "audio"
CHECKPOINT_PATH = EXPERIMENT_DIR / "models" / "checkpoints" / "BEATs_iter3_plus_AS2M.pt"
OUTPUT_PATH = EXPERIMENT_DIR / "features" / "beats_features.csv"

SAMPLE_RATE = 16000


def load_model() -> BEATs:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def extract_embedding(model: BEATs, file_path: pathlib.Path) -> np.ndarray:
    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    audio = torch.from_numpy(signal).float().unsqueeze(0)
    padding_mask = torch.zeros(1, audio.shape[1], dtype=torch.bool)
    with torch.no_grad():
        frame_features = model.extract_features(audio, padding_mask=padding_mask)[0]
    # Collapse the time axis so each clip becomes one fixed-length embedding.
    return frame_features.mean(dim=1).squeeze(0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N clips (for a quick smoke test before a full run).",
    )
    args = parser.parse_args()

    model = load_model()
    metadata = pd.read_csv(METADATA_PATH)
    if args.limit:
        metadata = metadata.head(args.limit)

    rows = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting BEATs embeddings"):
        file_path = AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
        try:
            embedding = extract_embedding(model, file_path)
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        rows.append([row["slice_file_name"], row["fold"], row["classID"], row["class"], *embedding])

    n_features = len(rows[0]) - 4
    columns = ["slice_file_name", "fold", "classID", "class"] + [f"beats_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(rows, columns=columns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(features_df)} feature rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
