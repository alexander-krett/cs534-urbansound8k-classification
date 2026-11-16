"""Re-evaluate MFCC, CLAP, and BEATs on their common set of successfully processed
clips, so accuracy/precision/recall/F1 are computed on an identical held-out
population across all three (BEATs' patch-embedding conv cannot process 19
extremely short clips that the other two pipelines handle fine)."""

import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_SETS = {
    "MFCC": ("features/mfcc_features.csv", "mfcc_"),
    "CLAP": ("clap_experiment/features/clap_features.csv", "clap_"),
    "BEATs": ("beats_experiment/features/beats_features.csv", "beats_"),
}
OUTPUT_PATH = pathlib.Path("comparison_common_subset_results.txt")


def evaluate(df: pd.DataFrame, feature_cols: list) -> tuple:
    fold_accuracies = []
    all_true, all_pred = [], []

    for test_fold in sorted(df["fold"].unique()):
        train_df = df[df["fold"] != test_fold]
        test_df = df[df["fold"] == test_fold]

        X_train, y_train = train_df[feature_cols].values, train_df["class"].values
        X_test, y_test = test_df[feature_cols].values, test_df["class"].values

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        all_true.extend(y_test)
        all_pred.extend(y_pred)

    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    report = classification_report(all_true, all_pred)
    return mean_acc, std_acc, report


def main() -> None:
    dfs = {name: pd.read_csv(path) for name, (path, _) in FEATURE_SETS.items()}
    common_clips = set.intersection(*(set(df["slice_file_name"]) for df in dfs.values()))
    print(f"Common clips across all three feature sets: {len(common_clips)}")

    summary_lines = [f"Common clips across all three feature sets: {len(common_clips)}\n"]
    for name, (_, prefix) in FEATURE_SETS.items():
        df = dfs[name]
        df = df[df["slice_file_name"].isin(common_clips)].reset_index(drop=True)
        feature_cols = [c for c in df.columns if c.startswith(prefix)]

        mean_acc, std_acc, report = evaluate(df, feature_cols)

        print(f"\n=== {name} (on common {len(df)}-clip subset) ===")
        print(f"Mean 10-fold accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
        print(report)

        summary_lines.append(f"\n=== {name} (on common {len(df)}-clip subset) ===\n")
        summary_lines.append(f"Mean 10-fold accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        summary_lines.append(report)

    OUTPUT_PATH.write_text("".join(summary_lines))
    print(f"\nSaved combined results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
