"""Frozen Audio Foundation Model features for UrbanSound8K.

Weights and model dependencies are provisioned separately by the Slurm setup
workflow. Importing this module never downloads models.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer", "siren",
    "street_music",
]
_CLASS_DESCRIPTIONS = [
    "an air conditioner running", "a car horn honking", "children playing",
    "a dog barking", "drilling", "an engine idling", "a gunshot",
    "a jackhammer operating", "a siren sounding", "street music",
]
PROMPTS = [f"An audio recording of {description}." for description in _CLASS_DESCRIPTIONS]
CLAP_CHECKPOINT = "630k-audioset-best.pt"
BEATS_CHECKPOINT = "BEATs_iter3_plus_AS2M.pt"
BEATS_SOURCE_REVISION = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"
BEATS_SOURCE_DIR = "unilm-beats"
CLAP_DIM = 512
BEATS_DIM = 768
CLAP_RATE = 48_000
CLAP_SAMPLES = 480_000
BEATS_RATE = 16_000
PREPROCESSING = {
    "clap": {"sample_rate": CLAP_RATE, "mono": True, "samples": CLAP_SAMPLES,
             "length_policy": "floor-repeat then zero-pad short; truncate-long"},
    "beats": {"sample_rate": BEATS_RATE, "mono": True,
              "pooling": "mean over valid temporal frames", "minimum_samples": 3200,
              "short_policy": "zero-pad to 200ms, mask padded samples"},
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(checkpoint_dir: Path, filename: str) -> Path:
    path = checkpoint_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint missing: {path}")
    return path


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load and validate the official UrbanSound8K metadata and audio files."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "metadata" / "UrbanSound8K.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"UrbanSound8K metadata missing: {csv_path}")
    raw = pd.read_csv(csv_path)
    required = {"slice_file_name", "fold", "class", "classID"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")
    frame = raw[["slice_file_name", "fold", "class", "classID"]].copy()
    frame["slice_file_name"] = frame["slice_file_name"].astype(str)
    for column in ("fold", "classID"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain finite integers")
        frame[column] = values.astype(int)
    if frame.empty:
        raise ValueError("UrbanSound8K metadata contains no rows")
    if frame["slice_file_name"].str.contains(r"[/\\]", regex=True).any():
        raise ValueError("slice_file_name must contain a basename, not a path")
    if not frame["fold"].between(1, 10).all():
        raise ValueError("UrbanSound8K fold values must be between 1 and 10")
    if not frame["classID"].between(0, 9).all():
        raise ValueError("UrbanSound8K classID values must be between 0 and 9")
    expected = frame["classID"].map(dict(enumerate(CLASSES)))
    if not expected.equals(frame["class"].reset_index(drop=True)):
        # Compare by values as pandas may preserve a non-default input index.
        if expected.tolist() != frame["class"].tolist():
            raise ValueError("class and classID do not match the official UrbanSound8K mapping")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "sample_id", frame["fold"].astype(str).radd("fold") + "/" + frame["slice_file_name"])
    if frame["sample_id"].duplicated().any():
        raise ValueError("UrbanSound8K metadata contains duplicate fold/file sample IDs")
    paths = [data_dir / "audio" / f"fold{fold}" / name
             for fold, name in zip(frame["fold"], frame["slice_file_name"])]
    absent = [p for p in paths if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"Audio file missing for metadata row: {absent[0]}")
    return frame


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def fuse_embeddings(clap: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Concatenate separately L2-normalized CLAP and BEATs representations."""
    clap = np.asarray(clap)
    beats = np.asarray(beats)
    if clap.ndim != 2 or beats.ndim != 2 or len(clap) != len(beats):
        raise ValueError("clap and beats must be 2D arrays with the same row count")
    return np.concatenate((l2_normalize(clap), l2_normalize(beats)), axis=1)


def _audio(path: str, rate: int, target_length: int | None = None) -> np.ndarray:
    import librosa

    signal, _ = librosa.load(path, sr=rate, mono=True)
    signal = np.asarray(signal, dtype=np.float32)
    if target_length is not None:
        if signal.size == 0:
            signal = np.zeros(target_length, dtype=np.float32)
        elif signal.size < target_length:
            repeats = target_length // signal.size
            signal = np.pad(np.tile(signal, repeats), (0, target_length - repeats * signal.size))
        else:
            signal = signal[:target_length]
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Audio must contain finite nonempty samples: {path}")
    return signal


def _load_clap(checkpoint_dir: Path, device: str) -> Any:
    """Load official CLAP state with explicit legacy-checkpoint semantics.

    Mirrors upstream load_state_dict, including the Transformers >=4.31 buffer
    migration, while making weights_only explicit for PyTorch >=2.6.
    """
    import torch
    package = importlib.import_module("laion_clap")
    model = package.CLAP_Module(enable_fusion=False, device=device)
    checkpoint = torch.load(str(_checkpoint_path(checkpoint_dir, CLAP_CHECKPOINT)),
                            map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    state.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(state, strict=True)
    model.eval()
    model.model.to(device).requires_grad_(False)
    return model


def _load_models(checkpoint_dir: Path, device: str) -> tuple[Any, Any]:
    import torch

    clap = _load_clap(checkpoint_dir, device)

    beats_path = _checkpoint_path(checkpoint_dir, BEATS_CHECKPOINT)
    source_dir = checkpoint_dir / BEATS_SOURCE_DIR
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Pinned BEATs source not found: {source_dir}")
    revision_file = source_dir / "REVISION"
    if not revision_file.is_file() or revision_file.read_text().strip() != BEATS_SOURCE_REVISION:
        raise ValueError(f"BEATs source must be pinned at revision {BEATS_SOURCE_REVISION}")
    source_str = str(source_dir.resolve())
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    beats_module = importlib.import_module("BEATs")
    checkpoint = torch.load(str(beats_path), map_location="cpu", weights_only=False)
    config = beats_module.BEATsConfig(checkpoint["cfg"])
    if config.finetuned_model:
        raise ValueError("Expected self-supervised BEATs_iter3_plus_AS2M checkpoint, got fine-tuned model")
    beats = beats_module.BEATs(config)
    beats.load_state_dict(checkpoint["model"])
    beats.to(device).eval()
    beats.requires_grad_(False)
    return clap, beats


def _embed_audio(clap: Any, beats: Any, paths: list[str], device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    clap_rows: list[np.ndarray] = []
    beats_rows: list[np.ndarray] = []
    for index, path in enumerate(paths):
        if index % 100 == 0:
            print(f"Extracting AFM embeddings: {index}/{len(paths)}", flush=True)
        c_audio = _audio(path, CLAP_RATE, CLAP_SAMPLES)[None, :]
        waveform = _audio(path, BEATS_RATE)
        original_length = len(waveform)
        if original_length < 3200:
            waveform = np.pad(waveform, (0, 3200 - original_length))
        b_audio = torch.from_numpy(waveform).unsqueeze(0).to(device)
        padding = torch.arange(len(waveform), device=device)[None, :] >= original_length
        with torch.inference_mode():
            c = clap.get_audio_embedding_from_data(c_audio, use_tensor=False)
            b, feature_padding = beats.extract_features(b_audio, padding_mask=padding)
        c = np.asarray(c, dtype=np.float32).reshape(-1)
        valid = (~feature_padding).unsqueeze(-1) if feature_padding is not None else torch.ones_like(b[..., :1], dtype=torch.bool)
        if not valid.any():
            raise ValueError(f"BEATs produced no valid features for {path}")
        b = ((b.detach().float() * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)).cpu().numpy().reshape(-1)
        if c.shape != (CLAP_DIM,) or b.shape != (BEATS_DIM,):
            raise ValueError(f"Unexpected AFM dimensions for {path}: CLAP {c.shape}, BEATs {b.shape}")
        clap_rows.append(c)
        beats_rows.append(b)
    return (np.asarray(clap_rows, dtype=np.float32).reshape((-1, CLAP_DIM)),
            np.asarray(beats_rows, dtype=np.float32).reshape((-1, BEATS_DIM)))


def _metadata_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame[["sample_id", "slice_file_name", "fold", "class", "classID"]]
                          .to_json(orient="records").encode("utf-8")).hexdigest()


def _source_fingerprint(frame: pd.DataFrame, data_dir: Path) -> str:
    digest = hashlib.sha256()
    for fold, name in zip(frame["fold"], frame["slice_file_name"]):
        file = data_dir / "audio" / f"fold{fold}" / name
        stat = file.stat()
        digest.update(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns}:".encode())
        digest.update(_hash_file(file).encode())
    return digest.hexdigest()


def _signature(frame: pd.DataFrame, data_dir: Path, checkpoint_dir: Path,
               limit: int | None) -> dict[str, Any]:
    clap_path = _checkpoint_path(checkpoint_dir, CLAP_CHECKPOINT)
    beats_path = _checkpoint_path(checkpoint_dir, BEATS_CHECKPOINT)
    revision_file = checkpoint_dir / BEATS_SOURCE_DIR / "REVISION"
    if not revision_file.is_file() or revision_file.read_text().strip() != BEATS_SOURCE_REVISION:
        raise ValueError(f"BEATs source must be pinned at revision {BEATS_SOURCE_REVISION}")
    beats_source_hashes = {
        name: _hash_file(checkpoint_dir / BEATS_SOURCE_DIR / name)
        for name in ("BEATs.py", "backbone.py", "modules.py")
    }
    clap_spec = importlib.util.find_spec("laion_clap")
    if clap_spec is None or not clap_spec.submodule_search_locations:
        raise ImportError("laion_clap package is required for AFM features")
    clap_source_dir = Path(next(iter(clap_spec.submodule_search_locations)))
    clap_source_hashes = {
        str(path.relative_to(clap_source_dir)): _hash_file(path)
        for path in sorted(clap_source_dir.rglob("*.py"))
    }
    return {
        "metadata_sha256": _metadata_fingerprint(frame),
        "audio_sha256": _source_fingerprint(frame, data_dir),
        "checkpoints": {CLAP_CHECKPOINT: _hash_file(clap_path), BEATS_CHECKPOINT: _hash_file(beats_path)},
        "beats_source_revision": BEATS_SOURCE_REVISION,
        "beats_source_sha256": beats_source_hashes,
        "laion_clap_version": importlib.metadata.version("laion-clap"),
        "laion_clap_source_sha256": clap_source_hashes,
        "preprocessing": PREPROCESSING,
        "extractor_sha256": _hash_file(Path(__file__)),
        "dependency_versions": {name: importlib.metadata.version(name) for name in ("torch", "torchaudio", "librosa", "numpy")},
        "smoke_limit": limit,
    }


def _atomic_save(cache_path: Path, frame: pd.DataFrame, clap: np.ndarray,
                 beats: np.ndarray, signature: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{cache_path.name}.", suffix=".tmp", dir=cache_path.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as stream:
            np.savez_compressed(stream, metadata=frame[["sample_id", "slice_file_name", "fold", "class", "classID"]]
                                .to_json(orient="records"), clap=clap, beats=beats,
                                signature=json.dumps(signature, sort_keys=True))
        os.replace(temp_name, cache_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _read_cache(cache_path: Path, signature: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if json.loads(str(data["signature"].item())) != signature:
                return None
            cached_frame = pd.read_json(io.StringIO(str(data["metadata"].item())), orient="records")
            expected = frame[["sample_id", "slice_file_name", "fold", "class", "classID"]].reset_index(drop=True)
            if not cached_frame.equals(expected):
                return None
            clap = np.asarray(data["clap"], dtype=np.float32)
            beats = np.asarray(data["beats"], dtype=np.float32)
        if clap.shape != (len(frame), CLAP_DIM) or beats.shape != (len(frame), BEATS_DIM):
            return None
        if not np.isfinite(clap).all() or not np.isfinite(beats).all():
            return None
        return clap, beats
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def load_or_extract(data_dir: str | Path, cache_dir: str | Path,
                    checkpoint_dir: str | Path, device: str = "cuda",
                    smoke_limit: int | None = None,
                    allow_extract: bool = True) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Load cached AFM vectors or extract aligned CLAP and BEATs features."""
    frame = load_metadata(data_dir)
    if smoke_limit is not None:
        if smoke_limit < 1:
            raise ValueError("smoke_limit must be a positive integer")
        frame = frame.head(smoke_limit).copy().reset_index(drop=True)
    checkpoint_dir = Path(checkpoint_dir)
    data_dir = Path(data_dir)
    signature = _signature(frame, data_dir, checkpoint_dir, smoke_limit)
    cache_path = Path(cache_dir) / "afm_features.npz"
    cached = _read_cache(cache_path, signature, frame)
    if cached is None:
        if not allow_extract:
            raise RuntimeError(f"AFM cache is missing or stale and extraction is disabled: {cache_path}")
        clap, beats = _load_models(checkpoint_dir, device)
        paths = [str(data_dir / "audio" / f"fold{fold}" / name)
                 for fold, name in zip(frame["fold"], frame["slice_file_name"])]
        clap_values, beats_values = _embed_audio(clap, beats, paths, device)
        if (clap_values.shape != (len(frame), CLAP_DIM) or beats_values.shape != (len(frame), BEATS_DIM)
                or not np.isfinite(clap_values).all() or not np.isfinite(beats_values).all()):
            raise ValueError("Extracted embeddings have invalid shapes or nonfinite values")
        _atomic_save(cache_path, frame, clap_values, beats_values, signature)
    else:
        clap_values, beats_values = cached
    result_frame = frame.copy()
    provenance = {
        "models": {"clap": CLAP_CHECKPOINT, "beats": BEATS_CHECKPOINT},
        "checkpoint_sha256": signature["checkpoints"],
        "beats_source_revision": BEATS_SOURCE_REVISION,
        "preprocessing": PREPROCESSING,
        "cache": str(cache_path),
        "cache_hit": cached is not None,
        "signature": signature,
    }
    return result_frame, {"clap": clap_values, "beats": beats_values}, provenance


def text_prototypes(checkpoint_dir: str | Path, device: str = "cuda") -> tuple[np.ndarray, dict[str, Any]]:
    """Return one CLAP text embedding per official class prompt, in class order."""
    checkpoint_dir = Path(checkpoint_dir)
    clap_path = _checkpoint_path(checkpoint_dir, CLAP_CHECKPOINT)
    import torch

    model = _load_clap(checkpoint_dir, device)
    with torch.inference_mode():
        vectors = model.get_text_embedding(PROMPTS, use_tensor=True)
    vectors = vectors.detach().float().cpu().numpy()
    if vectors.shape != (len(CLASSES), CLAP_DIM):
        raise ValueError(f"Unexpected CLAP text prototype shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("CLAP text prototypes must be finite")
    return l2_normalize(vectors), {
        "model": CLAP_CHECKPOINT,
        "checkpoint_sha256": _hash_file(clap_path),
        "prompts": list(PROMPTS),
        "classes": list(CLASSES),
    }
