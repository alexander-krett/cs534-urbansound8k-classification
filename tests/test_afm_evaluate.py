"""Synthetic checks for frozen-embedding evaluation; no audio/model downloads."""
import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import afm_evaluate as evaluate


class AfmEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "slice_file_name": [f"sample_{fold}_{i}.wav" for fold in (1, 2, 3) for i in range(10)],
            "fold": np.repeat([1, 2, 3], 10), "classID": np.tile(np.arange(10), 3),
        })
        rng = np.random.default_rng(4)
        self.clap = rng.normal(size=(30, 512)).astype(np.float32)
        self.beats = rng.normal(size=(30, 768)).astype(np.float32)
        self.proto = rng.normal(size=(10, 512)).astype(np.float32)

    def test_svm_is_fit_without_test_fold_and_metrics_are_complete(self):
        captured = {}
        class Spy:
            def fit(self, x, y):
                captured["x"], captured["y"] = x.copy(), y.copy()
                return self
            def predict(self, x):
                return self.labels[x[:, 0].astype(int) % 10]
        # Derive deterministic predictions from the original metadata row encoded below.
        spy = Spy()
        spy.labels = np.arange(10)
        def pipeline(*args):
            return spy
        x = self.clap.copy()
        x[:, 0] = np.arange(len(x))
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(evaluate, "make_pipeline", side_effect=pipeline), \
             mock.patch.object(evaluate, "_save_checkpoint"):
            report = evaluate._evaluate("clap_svm", x, self.frame, Path(tmp), {},
                                        122333, 1, 4, [2])
            predictions = pd.read_csv(Path(tmp) / "clap_svm_predictions.csv")
            self.assertTrue((predictions.fold == 2).all())
            self.assertEqual(predictions.slice_file_name.tolist(),
                             self.frame.loc[self.frame.fold == 2, "slice_file_name"].tolist())
        np.testing.assert_array_equal(captured["x"][:, 0], np.r_[0:10, 20:30])
        self.assertEqual(len(captured["y"]), 20)
        self.assertEqual(list(report["folds"]), ["2"])
        self.assertEqual(len(report["pooled"]["confusion_matrix"]), 10)
        self.assertEqual(set(report["fold_mean_std"]), set(evaluate.METRIC_NAMES))

    def test_full_interface_writes_artifacts_and_fused_embeddings_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            def fake_eval(name, x, frame, output, provenance, seed, epochs, batch_size,
                          selected_folds, prototypes=None):
                called[name] = x.shape[1]
                if name == "fused_mlp":
                    np.testing.assert_allclose(np.linalg.norm(x[:, :512], axis=1), 1., atol=1e-5)
                    np.testing.assert_allclose(np.linalg.norm(x[:, 512:], axis=1), 1., atol=1e-5)
                report = {"folds": {}, "fold_mean_std": {}, "pooled": {
                    "accuracy": .5, "macro_f1": .4, "weighted_f1": .5, "macro_precision": .4}}
                return report
            called = {}
            with mock.patch.object(evaluate, "_evaluate", side_effect=fake_eval), \
                 mock.patch.object(evaluate, "_load_compatible_baselines", return_value={"svm": {}, "cnn": {}}):
                result = evaluate.evaluate_all(self.frame, {"clap": self.clap, "beats": self.beats},
                                               self.proto, out, {"dataset": "synthetic"},
                                               selected_folds=[1])
            self.assertEqual(set(result), {"clap_zero_shot", "clap_svm", "clap_mlp", "beats_svm",
                                           "beats_mlp", "fused_svm", "fused_mlp", "baselines"})
            self.assertEqual(called, {"clap_zero_shot": 512, "clap_svm": 512, "clap_mlp": 512,
                                      "beats_svm": 768, "beats_mlp": 768,
                                      "fused_svm": 1280, "fused_mlp": 1280})
            self.assertTrue((out / "afm_results.tex").is_file())
            self.assertTrue((out / "afm_results.md").is_file())
            self.assertTrue((out / "afm_manifest.json").is_file())

    def test_baseline_verifies_rows_independent_of_order_and_rejects_wrong_metrics(self):
        frame = pd.DataFrame({"slice_file_name": [f"{i}.wav" for i in range(10)],
                              "fold": np.arange(1, 11), "classID": np.arange(10)})
        predictions = frame[["slice_file_name", "fold"]].copy()
        predictions["true_label"] = evaluate.CLASS_NAMES
        predictions["predicted_label"] = evaluate.CLASS_NAMES
        stats = evaluate._metrics(np.arange(10), np.arange(10))
        data = {"config": {"seed": 122333, "test_folds": list(range(1, 11)),
                           "source_sha256": hashlib.sha256((ROOT / "src/train_baselines.py").read_bytes()).hexdigest()},
                "labels": evaluate.CLASS_NAMES, "folds": {str(i): {} for i in range(1, 11)},
                "fold_mean_std": {}, "pooled": stats}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "svm_metrics.json"
            path.write_text(json.dumps(data))
            predictions.iloc[::-1].to_csv(directory / "svm_predictions.csv", index=False)
            found = evaluate._load_compatible_baselines(directory, {}, frame)
            self.assertEqual(found["svm"]["status"], "available")
            self.assertEqual(found["cnn"]["status"], "unavailable")
            data["pooled"]["accuracy"] = .1
            path.write_text(json.dumps(data))
            found = evaluate._load_compatible_baselines(directory, {}, frame)
            self.assertEqual(found["svm"]["status"], "unavailable")

    def test_mlp_accepts_all_three_embedding_dimensions(self):
        import torch
        for width in (512, 768, 1280):
            model = evaluate._mlp_class(torch, width)().eval()
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.zeros(2, width)).shape), (2, 10))

    def test_zero_shot_uses_cosine_similarity(self):
        prototypes = np.eye(10, dtype=np.float32)
        audio = prototypes[[4]] * 9
        np.testing.assert_array_equal(evaluate._zero_shot_predict(audio, prototypes), [4])

    def test_mlp_selects_epochs_without_test_data_then_refits_on_all_train_folds(self):
        fits, calls = [], []
        class Scaler:
            def fit(self, x):
                fits.append(x.copy())
                self.mean_ = np.zeros(x.shape[1])
                self.scale_ = np.ones(x.shape[1])
                return self
            def transform(self, x):
                return x
        class Model:
            pass
        def fit_mlp(x, y, seed, epochs, batch_size, validation=None):
            calls.append((x.copy(), y.copy(), epochs, validation))
            return Model(), 7
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(evaluate, "StandardScaler", side_effect=Scaler), \
             mock.patch.object(evaluate, "_fit_mlp", side_effect=fit_mlp), \
             mock.patch.object(evaluate, "_predict_mlp", return_value=np.zeros(10, dtype=int)), \
             mock.patch.object(evaluate, "_save_checkpoint"), \
             mock.patch.object(evaluate, "_write_artifacts"):
            report = evaluate._evaluate("clap_mlp", self.clap, self.frame, Path(tmp), {},
                                        122333, 20, 4, [1])
        # Fold 1 test; cyclic validation is fold 2; fold 3 is the selection fit set.
        np.testing.assert_array_equal(fits[0], self.clap[20:])
        np.testing.assert_array_equal(calls[0][0], self.clap[20:])
        np.testing.assert_array_equal(calls[0][3][0], self.clap[10:20])
        # Final scaler and model include both non-test folds, and use selected epoch count.
        np.testing.assert_array_equal(fits[1], self.clap[10:])
        np.testing.assert_array_equal(calls[1][0], self.clap[10:])
        self.assertEqual(calls[1][2], 7)
        self.assertEqual(report["folds"]["1"]["selected_epochs"], 7)


if __name__ == "__main__":
    unittest.main()
