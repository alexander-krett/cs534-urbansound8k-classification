"""Compare SVMs on the clips shared by three extracted feature tables."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_common_features(paths):
    tables = {}
    common_clips = None
    identity = ["slice_file_name", "fold", "class"]
    for name, path in paths.items():
        frame = pd.read_csv(path)
        if not set(identity).issubset(frame.columns):
            raise ValueError(f"{path}: expected columns {', '.join(identity)}")
        if frame[identity].isna().any().any() or frame["slice_file_name"].duplicated().any():
            raise ValueError(f"{path}: missing clip metadata or duplicate clip names")
        if not frame["fold"].isin(range(1, 11)).all():
            raise ValueError(f"{path}: expected fold numbers from 1 to 10")

        columns = []
        for column in frame.columns:
            if column.startswith(f"{name}_"):
                columns.append(column)
        if not columns:
            raise ValueError(f"{path}: no {name}_ feature columns")
        tables[name] = (frame, columns)
        clips = set(frame["slice_file_name"])
        common_clips = clips if common_clips is None else common_clips & clips

    if not common_clips:
        raise ValueError("The feature tables have no clips in common")

    reference = None
    for name, (frame, columns) in tables.items():
        frame = frame.loc[frame["slice_file_name"].isin(common_clips)].copy()
        metadata = frame[identity].set_index("slice_file_name").sort_index()
        if reference is None:
            reference = metadata
        elif not metadata.equals(reference):
            raise ValueError(f"{name}: fold or class mismatch for a shared clip")
        tables[name] = (frame, columns)

    if set(reference["fold"]) != set(range(1, 11)):
        raise ValueError("The shared clips must cover all ten official folds")
    return tables


def evaluate(frame, columns):
    fold_accuracies = {}
    predictions = []
    for fold in sorted(frame["fold"].unique()):
        train = frame.loc[frame["fold"] != fold]
        test = frame.loc[frame["fold"] == fold]
        pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
        pipeline.fit(train[columns].to_numpy(), train["class"].to_numpy())
        predicted = pipeline.predict(test[columns].to_numpy())
        fold_accuracies[str(fold)] = accuracy_score(test["class"], predicted)
        held_out = test[["slice_file_name", "fold"]].copy()
        held_out["true_label"] = test["class"]
        held_out["predicted_label"] = predicted
        predictions.append(held_out)

    predictions = pd.concat(predictions, ignore_index=True)
    accuracies = list(fold_accuracies.values())
    report = {
        "clips": len(predictions),
        "fold_accuracies": fold_accuracies,
        "mean_fold_accuracy": float(np.mean(accuracies)),
        "std_fold_accuracy": float(np.std(accuracies)),
        "pooled": classification_report(
            predictions["true_label"], predictions["predicted_label"],
            output_dict=True, zero_division=0,
        ),
    }
    return predictions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mfcc", type=Path, required=True)
    parser.add_argument("--clap", type=Path, required=True)
    parser.add_argument("--beats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tables = load_common_features({"mfcc": args.mfcc, "clap": args.clap, "beats": args.beats})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.log").open("w") as log_file:
        def log(message):
            print(message)
            print(message, file=log_file, flush=True)

        log(f"Common clips across all three feature sets: {len(tables['mfcc'][0])}")
        for name, (frame, columns) in tables.items():
            predictions, report = evaluate(frame, columns)
            predictions.to_csv(args.output_dir / f"{name}_svm_predictions.csv", index=False)
            labels = sorted(frame["class"].unique())
            display = ConfusionMatrixDisplay.from_predictions(
                predictions["true_label"], predictions["predicted_label"], labels=labels,
                xticks_rotation="vertical", colorbar=False,
            )
            report["labels"] = labels
            report["confusion_matrix"] = display.confusion_matrix.tolist()
            (args.output_dir / f"{name}_svm_metrics.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8",
            )
            display.figure_.set_size_inches(10, 10)
            display.figure_.tight_layout()
            display.figure_.savefig(args.output_dir / f"{name}_svm_confusion_matrix.png")
            plt.close(display.figure_)
            log(f"{name}: {report['mean_fold_accuracy']:.4f} +/- {report['std_fold_accuracy']:.4f}")


if __name__ == "__main__":
    main()
