"""Evaluate frozen CLAP and BEATs embeddings on official UrbanSound8K folds.

This module intentionally has no dependency on the audio feature extraction or
baseline training modules.  Torch and plotting libraries are imported lazily.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASS_IDS = list(range(10))
CLASS_NAMES = ["air_conditioner", "car_horn", "children_playing", "dog_bark",
               "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
METRIC_NAMES = ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_IDS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_precision": float(np.average(p, weights=support)) if support.sum() else 0.,
        "weighted_recall": float(np.average(r, weights=support)) if support.sum() else 0.,
        "weighted_f1": float(np.average(f, weights=support)) if support.sum() else 0.,
        "per_class": {CLASS_NAMES[i]: {"classID": i, "precision": float(p[i]), "recall": float(r[i]),
                                "f1": float(f[i]), "support": int(support[i])}
                      for i in CLASS_IDS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_IDS).tolist(),
    }


def _unit_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _fuse(clap: np.ndarray, beats: np.ndarray) -> np.ndarray:
    if clap.ndim != 2 or beats.ndim != 2 or clap.shape[0] != beats.shape[0]:
        raise ValueError("CLAP and BEATs embeddings must be aligned 2D arrays")
    return np.concatenate((_unit_rows(clap), _unit_rows(beats)), axis=1).astype(np.float32)


def _seed(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))


def _mlp_class(torch: Any, input_dim: int) -> Any:
    from torch import nn
    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(.3),
                                     nn.Linear(256, 10))
        def forward(self, x: Any) -> Any:
            return self.net(x)
    return MLP


def _fit_mlp(x: np.ndarray, y: np.ndarray, seed: int, epochs: int,
             batch_size: int, validation: tuple[np.ndarray, np.ndarray] | None = None,
             device_name: str | None = None
             ) -> tuple[Any, int]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    _seed(seed, torch)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model = _mlp_class(torch, x.shape[1])().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    data = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_loss, best_epoch, stale = float("inf"), 1, 0
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        if validation is not None:
            model.eval()
            vx, vy = validation
            with torch.no_grad():
                val_loss = sum(float(loss_fn(model(torch.as_tensor(vx[i:i + batch_size], dtype=torch.float32, device=device)),
                                            torch.as_tensor(vy[i:i + batch_size], dtype=torch.long, device=device))) * len(vx[i:i + batch_size])
                                for i in range(0, len(vx), batch_size)) / len(vx)
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
                if stale >= 10:
                    break
        else:
            best_epoch = epoch
    return model, best_epoch


def _predict_mlp(model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
            out.extend(model(xb).argmax(1).cpu().numpy().tolist())
    return np.asarray(out, dtype=np.int64)


def _save_checkpoint(path: Path, model_name: str, fold: int, artifact: Any,
                     labels: list[str], provenance: dict, seed: int,
                     selected_epochs: int | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    base = {"model": model_name, "test_fold": fold, "labels": labels,
            "provenance": _jsonable(provenance), "seed": seed}
    if model_name.endswith("_svm"):
        import joblib
        joblib.dump({**base, "estimator": artifact}, path / f"{model_name}_fold{fold}.joblib")
    else:
        import torch
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        scaler = artifact.get("scaler") if isinstance(artifact, dict) else None
        torch.save({**base, "state_dict": model.state_dict(),
                    "selected_epochs": selected_epochs,
                    "config": {"input_dim": model.net[0].in_features, "hidden_dim": 256, "dropout": .3,
                               "optimizer": "Adam", "learning_rate": .001,
                               "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
                               "scaler_scale": scaler.scale_.tolist() if scaler is not None else None}},
                   path / f"{model_name}_fold{fold}.pt")


def _write_artifacts(output: Path, name: str, frame: pd.DataFrame, indices: list[int],
                     truth: np.ndarray, pred: np.ndarray, labels: list[str],
                     report: dict, provenance: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = frame.iloc[indices].copy()
    rows["true_classID"] = truth
    rows["predicted_classID"] = pred
    rows["true_label"] = [labels[int(i)] for i in truth]
    rows["predicted_label"] = [labels[int(i)] for i in pred]
    rows.to_csv(output / f"{name}_predictions.csv", index=False)
    (output / f"{name}_metrics.json").write_text(json.dumps(
        {"labels": labels, "provenance": _jsonable(provenance), **_jsonable(report)}, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=np.asarray(report["pooled"]["confusion_matrix"]),
                                  display_labels=labels)
    disp.plot(xticks_rotation="vertical", cmap="Blues", colorbar=False)
    disp.figure_.set_size_inches(10, 9)
    disp.figure_.tight_layout()
    disp.figure_.savefig(output / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(disp.figure_)


def _evaluate(name: str, x: np.ndarray, frame: pd.DataFrame, output: Path,
              provenance: dict, seed: int, epochs: int, batch_size: int,
              selected_folds: list[int] | None, prototypes: np.ndarray | None = None) -> dict:
    folds = frame["fold"].to_numpy(dtype=int)
    y = frame["classID"].to_numpy(dtype=int)
    labels = CLASS_NAMES
    available = sorted(np.unique(folds).tolist())
    if selected_folds is not None:
        unknown = sorted(set(selected_folds) - set(available))
        if unknown:
            raise ValueError(f"Requested folds are absent from metadata: {unknown}")
        available = [f for f in available if f in selected_folds]
    if not available:
        raise ValueError("At least one test fold must be selected")
    true_all, pred_all, index_all = [], [], []
    fold_report, fold_epochs = {}, {}
    device_name = (provenance.get("run", {}) or {}).get("device")
    for fold in available:
        print(f"[{name}] fold {fold}: starting", flush=True)
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if not len(train) or not len(test):
            raise ValueError(f"Fold {fold} needs nonempty train and test sets")
        selected_epoch = None
        if name == "clap_zero_shot":
            pred = _zero_shot_predict(x[test], prototypes)
            artifact = None
        elif name.endswith("_svm"):
            artifact = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
            artifact.fit(x[train], y[train])
            pred = artifact.predict(x[test])
        else:
            import torch
            validation_fold = (fold % 10) + 1
            fit = np.flatnonzero((folds != fold) & (folds != validation_fold))
            val = np.flatnonzero(folds == validation_fold)
            if not len(fit) or not len(val):
                raise ValueError(f"MLP fold {fold} requires cyclic validation fold {validation_fold}")
            selection_scaler = StandardScaler().fit(x[fit])
            fit_args = ((selection_scaler.transform(x[fit]), y[fit], seed, epochs, batch_size,
                         (selection_scaler.transform(x[val]), y[val])))
            selection_model, selected_epoch = (_fit_mlp(*fit_args, device_name=device_name)
                                               if device_name else _fit_mlp(*fit_args))
            del selection_model
            scaler = StandardScaler().fit(x[train])
            fit_args = (scaler.transform(x[train]), y[train], seed, selected_epoch, batch_size)
            model, _ = (_fit_mlp(*fit_args, device_name=device_name)
                        if device_name else _fit_mlp(*fit_args))
            pred = _predict_mlp(model, scaler.transform(x[test]), batch_size)
            artifact = {"model": model, "scaler": scaler}
            fold_epochs[str(fold)] = selected_epoch
        fold_report[str(fold)] = _metrics(y[test], pred)
        if selected_epoch is not None:
            fold_report[str(fold)]["selected_epochs"] = selected_epoch
        true_all.extend(y[test].tolist())
        pred_all.extend(np.asarray(pred, dtype=int).tolist())
        index_all.extend(test.tolist())
        if name != "clap_zero_shot":
            _save_checkpoint(output, name, fold, artifact, labels, provenance, seed, selected_epoch)
        print(f"[{name}] fold {fold}: complete", flush=True)
    truth, pred = np.asarray(true_all), np.asarray(pred_all)
    report = {"folds": fold_report,
              "fold_mean_std": {metric: {"mean": float(np.mean([v[metric] for v in fold_report.values()])),
                                           "std": float(np.std([v[metric] for v in fold_report.values()]))}
                                for metric in METRIC_NAMES},
              "pooled": _metrics(truth, pred)}
    if fold_epochs:
        report["selected_epochs"] = fold_epochs
    _write_artifacts(output, name, frame, index_all, truth, pred, labels, report, provenance)
    return report


def _zero_shot_predict(audio: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    if audio.ndim != 2 or prototypes.ndim != 2 or prototypes.shape[0] != 10:
        raise ValueError("zero-shot evaluation requires 10 prototype vectors")
    return (_unit_rows(audio) @ _unit_rows(prototypes).T).argmax(axis=1).astype(int)


def evaluate_all(frame: pd.DataFrame, embeddings: dict[str, np.ndarray], prototypes: np.ndarray,
                 output_dir: Path, provenance: dict, seed: int = 122333,
                 epochs: int = 100, batch_size: int = 64,
                 selected_folds: list[int] | None = None,
                 baseline_dir: Path = Path("models")) -> dict:
    """Evaluate CLAP zero-shot and SVM/MLP on CLAP, BEATs, and fusion.

    ``frame`` must align row-for-row with each embedding matrix and contain the
    official UrbanSound8K ``fold`` and ``classID`` columns.
    """
    required = {"fold", "classID"}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame needs columns {sorted(required)}")
    if "slice_file_name" not in frame.columns:
        raise ValueError("frame needs official slice_file_name values for auditable predictions")
    if not {"clap", "beats"}.issubset(embeddings):
        raise ValueError("embeddings must contain clap and beats matrices")
    clap = np.asarray(embeddings["clap"], dtype=np.float32)
    beats = np.asarray(embeddings["beats"], dtype=np.float32)
    if clap.ndim != 2 or beats.ndim != 2 or len(clap) != len(frame) or len(beats) != len(frame):
        raise ValueError("embedding rows must align with frame")
    if clap.shape[1] != 512 or beats.shape[1] != 768:
        raise ValueError("expected CLAP 512 and BEATs 768 embedding dimensions")
    if not np.isfinite(clap).all() or not np.isfinite(beats).all():
        raise ValueError("embedding arrays must contain only finite values")
    ids = frame["classID"].to_numpy(dtype=int)
    if not np.isin(ids, CLASS_IDS).all():
        raise ValueError("classID values must be official IDs 0..9")
    proto = np.asarray(prototypes, dtype=np.float32)
    if proto.shape != (10, clap.shape[1]):
        raise ValueError(f"prototypes must have shape (10, {clap.shape[1]})")
    if not np.isfinite(proto).all() or np.any(np.linalg.norm(proto, axis=1) == 0):
        raise ValueError("prototypes must be finite nonzero vectors")
    folds = frame["fold"].to_numpy(dtype=int)
    if not np.isin(folds, np.arange(1, 11)).all():
        raise ValueError("fold values must be official IDs 1..10")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if selected_folds is None and set(np.unique(folds)) != set(range(1, 11)):
        raise ValueError("full evaluation requires all ten official folds")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(provenance)
    run_config = dict(provenance.get("run", {}))
    run_config.update({"seed": seed, "epochs_max": epochs, "batch_size": batch_size})
    run_config.setdefault("device", "cuda" if _cuda_available() else "cpu")
    provenance["run"] = run_config
    fused = _fuse(clap, beats)
    reports = {}
    reports["clap_zero_shot"] = _evaluate("clap_zero_shot", clap, frame, output,
                                          provenance, seed, epochs, batch_size, selected_folds, proto)
    for modality, matrix in (("clap", clap), ("beats", beats), ("fused", fused)):
        for classifier in ("svm", "mlp"):
            name = f"{modality}_{classifier}"
            reports[name] = _evaluate(name, matrix, frame, output, provenance,
                                      seed, epochs, batch_size, selected_folds)
    reports["baselines"] = _load_compatible_baselines(Path(baseline_dir), provenance, frame)
    _write_latex_table(output / "afm_results.tex", reports)
    _write_markdown_table(output / "afm_results.md", reports)
    (output / "afm_manifest.json").write_text(json.dumps({"provenance": _jsonable(provenance),
        "models": list(reports), "baselines": reports["baselines"]}, indent=2) + "\n")
    return reports


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _load_compatible_baselines(directory: Path, provenance: dict,
                               frame: pd.DataFrame | None = None) -> dict:
    """Load full-fold baselines only when code, seed, labels and prediction rows match."""
    result = {}
    source = Path(__file__).with_name("train_baselines.py")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    for name in ("svm", "cnn"):
        path = directory / f"{name}_metrics.json"
        if not path.is_file():
            result[name] = {"status": "unavailable", "reason": "metrics manifest missing"}
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            result[name] = {"status": "unavailable", "reason": "invalid metrics manifest"}
            continue
        config = data.get("config", {})
        predictions_path = directory / f"{name}_predictions.csv"
        try:
            compatible = (set(map(int, config.get("test_folds", []))) == set(range(1, 11))
                          and set(data.get("folds", {})) == {str(i) for i in range(1, 11)}
                          and config.get("seed") == provenance.get("run", {}).get("seed", 122333)
                          and config.get("source_sha256") == expected_hash
                          and data.get("labels") == CLASS_NAMES
                          and all(key in data for key in ("folds", "fold_mean_std", "pooled"))
                          and predictions_path.is_file() and frame is not None)
            if compatible:
                pred = pd.read_csv(predictions_path)
                expected = frame[["slice_file_name", "fold", "classID"]].rename(
                    columns={"classID": "true_label"}).copy()
                expected["true_label"] = expected["true_label"].map(lambda x: CLASS_NAMES[int(x)])
                columns = ["slice_file_name", "fold", "true_label"]
                compatible = (len(pred) == len(expected)
                              and not pred.duplicated(["slice_file_name", "fold"]).any()
                              and pred["predicted_label"].isin(CLASS_NAMES).all()
                              and pred[columns].sort_values(columns[:2]).reset_index(drop=True).equals(
                                  expected.sort_values(columns[:2]).reset_index(drop=True)))
                if compatible:
                    lookup = {label: i for i, label in enumerate(CLASS_NAMES)}
                    recomputed = _metrics(pred.true_label.map(lookup).to_numpy(), pred.predicted_label.map(lookup).to_numpy())
                    compatible = all(np.isclose(recomputed[key], data["pooled"][key])
                                     for key in ("accuracy", "macro_f1", "weighted_f1"))
        except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError):
            compatible = False
        result[name] = ({"status": "available", "metrics": data} if compatible else
                        {"status": "unavailable", "reason": "10-fold provenance mismatch"})
    return result


def _write_latex_table(path: Path, reports: dict) -> None:
    rows = [r"\begin{tabular}{lrrrr}", r"Model & Accuracy & Macro F1 & Weighted F1 & Macro Precision \\",
            r"\hline"]
    for name in ("clap_zero_shot", "clap_svm", "clap_mlp", "beats_svm", "beats_mlp",
                 "fused_svm", "fused_mlp"):
        escaped_name = name.replace("_", r"\_")
        stats = reports[name]["fold_mean_std"]
        rows.append(f"{escaped_name} & {_mean_std(stats, 'accuracy')} & {_mean_std(stats, 'macro_f1')} & "
                    f"{_mean_std(stats, 'weighted_f1')} & {_mean_std(stats, 'macro_precision')} \\\\")
    for name, entry in reports.get("baselines", {}).items():
        if entry.get("status") == "available":
            stats = entry["metrics"]["fold_mean_std"]
            rows.append(f"{name.upper()} baseline & {_mean_std(stats, 'accuracy')} & {_mean_std(stats, 'macro_f1')} & "
                        f"{_mean_std(stats, 'weighted_f1')} & -- \\\\")
        else:
            rows.append(f"{name.upper()} baseline (unavailable) & -- & -- & -- & -- \\\\")
    rows.append(r"\end{tabular}")
    path.write_text("\n".join(rows) + "\n")


def _mean_std(stats: dict, metric: str) -> str:
    item = stats.get(metric)
    return f"{item['mean']:.3f} $\\pm$ {item['std']:.3f}" if item else "--"


def _write_markdown_table(path: Path, reports: dict) -> None:
    lines = ["| Model | Accuracy | Macro F1 | Weighted F1 | Macro precision |",
             "|---|---:|---:|---:|---:|"]
    for name in ("clap_zero_shot", "clap_svm", "clap_mlp", "beats_svm", "beats_mlp", "fused_svm", "fused_mlp"):
        lines.append(f"| {name} | {_mean_std(reports[name]['fold_mean_std'], 'accuracy')} | "
                     f"{_mean_std(reports[name]['fold_mean_std'], 'macro_f1')} | "
                     f"{_mean_std(reports[name]['fold_mean_std'], 'weighted_f1')} | "
                     f"{_mean_std(reports[name]['fold_mean_std'], 'macro_precision')} |")
    for name, entry in reports.get("baselines", {}).items():
        if entry.get("status") == "available":
            stats = entry["metrics"]["fold_mean_std"]
            cells = [_mean_std(stats, key) for key in ("accuracy", "macro_f1", "weighted_f1", "macro_precision")]
            lines.append(f"| {name.upper()} baseline | " + " | ".join(cells) + " |")
        else:
            lines.append(f"| {name.upper()} baseline (unavailable) | — | — | — | — |")
    lines += ["", "Fold mean ± population standard deviation across evaluated folds.",
              "Pooled sample metrics are recorded in each model's metrics JSON."]
    path.write_text("\n".join(lines) + "\n")
