"""Extract CLAP audio embeddings from UrbanSound8K audio into a single feature table."""

import argparse
import pathlib

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

EXPERIMENT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = pathlib.Path("data/UrbanSound8K")
METADATA_PATH = DATA_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_PATH = EXPERIMENT_DIR / "features" / "clap_features.csv"

MODEL_NAME = "laion/clap-htsat-unfused"


def load_model():
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def extract_embedding(model: ClapModel, processor: ClapProcessor, file_path: pathlib.Path) -> np.ndarray:
    sample_rate = processor.feature_extractor.sampling_rate
    signal, _ = librosa.load(file_path, sr=sample_rate)
    inputs = processor(audio=signal, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        output = model.get_audio_features(**inputs)
    return output.pooler_output.squeeze(0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N clips (for a quick smoke test before a full run).",
    )
    args = parser.parse_args()

    model, processor = load_model()
    metadata = pd.read_csv(METADATA_PATH)
    if args.limit:
        metadata = metadata.head(args.limit)

    rows = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting CLAP embeddings"):
        file_path = AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
        try:
            embedding = extract_embedding(model, processor, file_path)
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        rows.append([row["slice_file_name"], row["fold"], row["classID"], row["class"], *embedding])

    n_features = len(rows[0]) - 4
    columns = ["slice_file_name", "fold", "classID", "class"] + [f"clap_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(rows, columns=columns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(features_df)} feature rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
