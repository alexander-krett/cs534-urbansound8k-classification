"""Train and evaluate an SVM classifier on extracted UrbanSound8K MFCC features.

Uses the dataset's official 10-fold split (never re-shuffle the folds: clips
from the same source recording are grouped together to prevent leakage).
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_PATH = pathlib.Path("features/mfcc_features.csv")
MODEL_PATH = pathlib.Path("models/svm_model.joblib")
CONFUSION_MATRIX_PATH = pathlib.Path("models/confusion_matrix.png")
RESULTS_PATH = pathlib.Path("models/results.txt")


def load_features():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c.startswith("mfcc_")]
    return df, feature_cols


def main() -> None:
    df, feature_cols = load_features()

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

        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)
        print(f"Fold {test_fold}: accuracy = {fold_acc:.4f}")

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    report = classification_report(all_true, all_pred)

    print(f"\nMean 10-fold accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print("\nOverall classification report:")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w") as f:
        f.write("Per-fold accuracy:\n")
        for fold, acc in zip(sorted(df["fold"].unique()), fold_accuracies):
            f.write(f"Fold {fold}: accuracy = {acc:.4f}\n")
        f.write(f"\nMean 10-fold accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        f.write("\nOverall classification report:\n")
        f.write(report)
    print(f"Saved results to {RESULTS_PATH}")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
    ])
    final_pipeline.fit(df[feature_cols].values, df["class"].values)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nSaved final model (trained on all folds) to {MODEL_PATH}")

    disp = ConfusionMatrixDisplay.from_predictions(all_true, all_pred, xticks_rotation="vertical")
    disp.figure_.set_size_inches(10, 10)
    disp.figure_.tight_layout()
    disp.figure_.savefig(CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
