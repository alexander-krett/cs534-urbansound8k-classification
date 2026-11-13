"""Reproducible fold-based MFCC/SVM and log-Mel/CNN UrbanSound8K baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import subprocess
import os
from typing import Any

import librosa
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 122333
SAMPLE_RATE = 22050
CNN_CONFIG = {"sample_rate": SAMPLE_RATE, "duration_seconds": 4, "n_mels": 64,
              "n_fft": 1024, "hop_length": 512, "db_range": [-80, 0], "frames": 173}
MFCC_CONFIG = {"sample_rate": SAMPLE_RATE, "n_mfcc": 40, "statistics": ["mean", "std"]}


TORCH_THREADS = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(TORCH_THREADS)


def mfcc_vector(path: pathlib.Path) -> np.ndarray:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1))).astype(np.float32)


def logmel_image(path: pathlib.Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    size = SAMPLE_RATE * 4
    if len(y) > size:
        start = (len(y) - size) // 2
        y = y[start:start + size]
    else:
        y = librosa.util.fix_length(y, size=size)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64,
                                         n_fft=1024, hop_length=512)
    db = (np.full_like(mel, -80., dtype=np.float32) if not np.any(mel > 0)
          else librosa.power_to_db(mel, ref=np.max, top_db=80.0))
    # Normalize the dynamic range [-80, 0] dB to [0, 1].
    db = np.clip(db, -80.0, 0.0)
    frames = 173
    if db.shape[1] >= frames:
        start = (db.shape[1] - frames) // 2
        db = db[:, start:start + frames]
    else:
        db = np.pad(db, ((0, 0), (0, frames - db.shape[1])), constant_values=-80.)
    return ((db + 80.) / 80.).astype(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {str(label): {"precision": float(pi), "recall": float(ri), "f1": float(fi),
                                    "support": int(n)}
                      for label, pi, ri, fi, n in zip(labels, p, r, f, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _cnn_class(torch: Any) -> Any:
    from torch import nn
    class CNN(nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            layers = []
            in_channels = 1
            for channels in (32, 64, 128, 256):
                layers.extend([nn.Conv2d(in_channels, channels, 3, padding=1),
                               nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)])
                in_channels = channels
            layers.extend([nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(.3),
                           nn.Linear(256, n_classes)])
            self.net = nn.Sequential(*layers)
        def forward(self, x: Any) -> Any:
            return self.net(x)
    return CNN


def _predict_batched(model: Any, X: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    import torch
    result = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size, None]).to(device)
            result.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.int64)


def _fit_cnn(X: np.ndarray, y: np.ndarray, labels: list[str], seed: int, epochs: int,
             batch_size: int, validation: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[Any, int]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_ids = {label: i for i, label in enumerate(labels)}
    targets = np.asarray([label_ids[label] for label in y], dtype=np.int64)
    model = _cnn_class(torch)(len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X[:, None]), torch.from_numpy(targets)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_loss, best_epoch, stale = float("inf"), 0, 0
    patience = 5
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        if validation is None:
            best_epoch = epoch
            continue
        model.eval()
        vx, vy = validation
        val_targets = np.asarray([label_ids[label] for label in vy], dtype=np.int64)
        total_loss, count = 0., 0
        with torch.no_grad():
            for start in range(0, len(vx), batch_size):
                stop = start + batch_size
                xb = torch.from_numpy(vx[start:stop, None]).to(device)
                yb = torch.from_numpy(val_targets[start:stop]).to(device)
                total_loss += float(loss_fn(model(xb), yb)) * len(xb)
                count += len(xb)
        val_loss = total_loss / max(1, count)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return model, max(1, best_epoch)


def _evaluate_cnn_fold(X: np.ndarray, y: np.ndarray, folds: np.ndarray, test_fold: int,
                       labels: list[str], seed: int, epochs: int, batch_size: int
                       ) -> tuple[np.ndarray, Any, int]:
    import torch
    validation_fold = (test_fold % 10) + 1
    train_idx = np.flatnonzero((folds != test_fold) & (folds != validation_fold))
    val_idx = np.flatnonzero(folds == validation_fold)
    test_idx = np.flatnonzero(folds == test_fold)
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError(f"fold {test_fold} requires nonempty fit, validation fold {validation_fold}, and test sets")
    selection_model, selected_epochs = _fit_cnn(X[train_idx], y[train_idx], labels, seed,
                                                 epochs, batch_size, (X[val_idx], y[val_idx]))
    del selection_model
    # Reinitialize identically, then fit on all nine non-test folds for the chosen
    # epoch count; the validation fold no longer remains excluded from training.
    model, _ = _fit_cnn(X[folds != test_fold], y[folds != test_fold], labels,
                         seed, selected_epochs, batch_size)
    device = next(model.parameters()).device
    encoded = _predict_batched(model, X[test_idx], device, batch_size)
    return np.asarray(labels)[encoded], model, selected_epochs


def evaluate_arrays(X: np.ndarray, y: np.ndarray, folds: np.ndarray, labels: list[str],
                    model_name: str = "svm", seed: int = SEED, epochs: int = 30,
                    batch_size: int = 64, selected_folds: list[int] | None = None
                    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    fold_numbers = sorted(np.unique(folds).astype(int).tolist())
    if selected_folds is not None:
        unknown = sorted(set(selected_folds) - set(fold_numbers))
        if unknown:
            raise ValueError(f"Requested folds are absent from the metadata: {unknown}")
        fold_numbers = [fold for fold in fold_numbers if fold in selected_folds]
    if not fold_numbers:
        raise ValueError("At least one test fold must be selected")
    all_true, all_pred, all_indices = [], [], []
    fold_reports, fold_accuracies, fold_macro_f1, fold_weighted_f1 = {}, [], [], []
    for fold in fold_numbers:
        test_idx = np.flatnonzero(folds == fold)
        if not np.any(folds != fold) or not len(test_idx):
            raise ValueError(f"Fold {fold} needs nonempty train and test sets")
        if model_name == "svm":
            estimator = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
            estimator.fit(X[folds != fold], y[folds != fold])
            pred = estimator.predict(X[test_idx])
            artifact = estimator
        elif model_name == "cnn":
            pred, artifact, selected_epochs = _evaluate_cnn_fold(
                X, y, folds, fold, labels, seed, epochs, batch_size)
        else:
            raise ValueError("model_name must be 'svm' or 'cnn'")
        fold_metric = metrics(y[test_idx], pred, labels)
        if model_name == "cnn":
            fold_metric["selected_epochs"] = selected_epochs
        fold_reports[str(fold)] = fold_metric
        fold_accuracies.append(fold_metric["accuracy"])
        fold_macro_f1.append(fold_metric["macro_f1"])
        fold_weighted_f1.append(fold_metric["weighted_f1"])
        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())
        all_indices.extend(test_idx.tolist())
        yield_artifact(artifact, model_name, fold, selected_epochs if model_name == "cnn" else None)
    report = {
        "folds": fold_reports,
        "fold_mean_std": {key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                           for key, values in (("accuracy", fold_accuracies), ("macro_f1", fold_macro_f1),
                                               ("weighted_f1", fold_weighted_f1))},
        "pooled": metrics(np.asarray(all_true), np.asarray(all_pred), labels),
    }
    return report, np.asarray(all_true), np.asarray(all_pred), np.asarray(all_indices)


def yield_artifact(artifact: Any, model_name: str, fold: int, selected_epochs: int | None) -> None:
    """Save fold checkpoint/model, keeping artifacts separate by test fold."""
    output = _OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{model_name}_fold{fold}.{'pt' if model_name == 'cnn' else 'joblib'}"
    if model_name == "svm":
        import joblib
        joblib.dump(artifact, path)
    else:
        import torch
        torch.save({"state_dict": artifact.cpu().state_dict(), "selected_epochs": selected_epochs}, path)


_OUTPUT_DIR = pathlib.Path("models")


def _cache_signature(metadata: pathlib.Path, model_name: str) -> str:
    config = MFCC_CONFIG if model_name == "svm" else CNN_CONFIG
    stat = metadata.stat()
    frame = pd.read_csv(metadata, usecols=["fold", "slice_file_name"])
    audio_root = metadata.parent.parent / "audio"
    sources = []
    for row in frame.itertuples(index=False):
        path = audio_root / f"fold{row.fold}" / row.slice_file_name
        try:
            audio_stat = path.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dataset audio file listed in metadata was not found: {path}") from exc
        sources.append((row.slice_file_name, int(row.fold), audio_stat.st_size, audio_stat.st_mtime_ns))
    source_hash = hashlib.sha256(json.dumps(sources, separators=(",", ":")).encode()).hexdigest()
    value = {"metadata": str(metadata.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
             "audio_sources": source_hash, "model": model_name, "config": config}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_or_create_data(data_dir: pathlib.Path, model_name: str, cache_path: pathlib.Path
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    metadata = data_dir / "metadata" / "UrbanSound8K.csv"
    signature = _cache_signature(metadata, model_name)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["signature"].item()) == signature:
                frame = pd.read_csv(metadata)
                labels = sorted(frame["class"].astype(str).unique().tolist())
                return cache["X"], cache["y"].astype(str), cache["folds"], labels, frame
    frame = pd.read_csv(metadata)
    required = {"fold", "slice_file_name", "class", "classID"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    extractor = mfcc_vector if model_name == "svm" else logmel_image
    audio = data_dir / "audio"
    first_row = next(frame.itertuples(index=False), None)
    if first_row is None:
        raise ValueError(f"Metadata contains no audio rows: {metadata}")
    first_path = audio / f"fold{first_row.fold}" / first_row.slice_file_name
    if not first_path.is_file():
        raise FileNotFoundError(f"Dataset audio file listed in metadata was not found: {first_path}")
    first_feature = extractor(first_path)
    X = np.empty((len(frame), *first_feature.shape), dtype=np.float32)
    X[0] = first_feature
    for index, row in enumerate(frame.iloc[1:].itertuples(index=False), start=1):
        path = audio / f"fold{row.fold}" / row.slice_file_name
        if not path.is_file():
            raise FileNotFoundError(f"Dataset audio file listed in metadata was not found: {path}")
        feature = extractor(path)
        if feature.shape != first_feature.shape:
            raise ValueError(f"Unexpected feature shape {feature.shape} for {path}")
        X[index] = feature
    y = np.asarray(frame["class"].astype(str), dtype=str)
    folds = frame["fold"].to_numpy(dtype=int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, folds=folds, signature=signature)
    labels = sorted(frame["class"].astype(str).unique().tolist())
    return X, y, folds, labels, frame


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("svm", "cnn", "both"), required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data/UrbanSound8K"))
    parser.add_argument("--cache", type=pathlib.Path, help="preprocessing cache; defaults to features/<model>_features.npz")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--folds", type=int, nargs="+", help="test fold(s); default evaluates all")
    parser.add_argument("--prepare-cache-only", action="store_true", help="extract/cache features and exit")
    args = parser.parse_args()
    global TORCH_THREADS
    TORCH_THREADS = max(1, args.torch_threads)
    if args.model == "both":
        if not args.prepare_cache_only:
            parser.error("--model both is only valid with --prepare-cache-only")
        for name in ("svm", "cnn"):
            path = (args.cache / f"{name}_features.npz") if args.cache else pathlib.Path("features") / f"{name}_features.npz"
            X, _, _, _, _ = load_or_create_data(args.data_dir, name, path)
            print(f"Prepared {name} cache ({X.shape}) at {path}")
        return
    global _OUTPUT_DIR
    _OUTPUT_DIR = args.output_dir
    cache = args.cache or pathlib.Path("features") / f"{args.model}_features.npz"
    X, y, folds, labels, frame = load_or_create_data(args.data_dir, args.model, cache)
    print(f"Loaded {len(y)} samples with shape {X.shape} from {cache}")
    if args.prepare_cache_only:
        return
    seed_everything(args.seed)
    report, truth, pred, row_indices = evaluate_arrays(X, y, folds, labels, args.model,
                                                        args.seed, args.epochs,
                                                        args.batch_size, args.folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = frame.iloc[row_indices][["slice_file_name", "fold"]].copy()
    predictions["true_label"] = truth
    predictions["predicted_label"] = pred
    predictions.to_csv(args.output_dir / f"{args.model}_predictions.csv", index=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or "unknown"
    source_path = pathlib.Path(__file__).resolve()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, check=False).stdout.strip()
    config = {"model": args.model, "seed": args.seed, "epochs_max": args.epochs,
              "batch_size": args.batch_size, "test_folds": args.folds or sorted(np.unique(folds).astype(int).tolist()),
              "architecture": "4x Conv3x3-BN-ReLU-MaxPool channels 32,64,128,256; GAP; Dropout 0.3; Linear-10" if args.model == "cnn" else "40 MFCC mean/std; train-fold scaler; RBF SVC C=10 gamma=scale",
              "feature_config": CNN_CONFIG if args.model == "cnn" else MFCC_CONFIG,
              "validation": "next official fold cyclically; reinitialize and fit all nine folds for selected epochs" if args.model == "cnn" else None,
              "code_revision": revision,
              "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "working_tree_dirty": bool(dirty),
              "versions": {"python": __import__("platform").python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "sklearn": sklearn.__version__,
                           "librosa": librosa.__version__, "torch": _package_version("torch")}}
    (args.output_dir / f"{args.model}_metrics.json").write_text(
        json.dumps({"config": config, "labels": labels, **report}, indent=2) + "\n")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay.from_predictions(truth, pred, labels=labels, xticks_rotation="vertical")
    disp.figure_.set_size_inches(10, 10)
    disp.figure_.tight_layout()
    disp.figure_.savefig(args.output_dir / f"{args.model}_confusion_matrix.png")
    plt.close(disp.figure_)
    print(json.dumps(report["fold_mean_std"], indent=2))
    print(f"Saved metrics, predictions, and checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()
