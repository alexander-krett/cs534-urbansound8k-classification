"""Small, dataset-independent end-to-end baseline smoke tests."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_baselines as baselines


class BaselineSmokeTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> None:
        audio = root / "data" / "UrbanSound8K" / "audio"
        meta = root / "data" / "UrbanSound8K" / "metadata"
        meta.mkdir(parents=True)
        rows = []
        rate = 22050
        time = np.arange(rate, dtype=np.float32) / rate
        for fold in range(1, 4):
            (audio / f"fold{fold}").mkdir(parents=True)
            for class_id, (label, freq) in enumerate((("low", 220.), ("high", 880.))):
                name = f"tone_{fold}_{label}.wav"
                signal = .5 * np.sin(2 * np.pi * freq * time).astype(np.float32)
                sf.write(audio / f"fold{fold}" / name, signal, rate)
                rows.append({"slice_file_name": name, "fold": fold,
                             "classID": class_id, "class": label})
        pd.DataFrame(rows).to_csv(meta / "UrbanSound8K.csv", index=False)

    def run_cli(self, root: Path, model: str) -> dict:
        data = root / "data" / "UrbanSound8K"
        output = root / "out"
        cache = root / f"{model}.npz"
        env = dict(os.environ, MPLBACKEND="Agg")
        result = subprocess.run([
            sys.executable, str(ROOT / "src" / "train_baselines.py"),
            "--model", model, "--data-dir", str(data), "--output-dir", str(output),
            "--cache", str(cache), "--folds", "1", "--epochs", "1",
            "--batch-size", "2", "--seed", "122333",
        ], cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        metrics = json.loads((output / f"{model}_metrics.json").read_text())
        self.assertEqual(list(metrics["folds"]), ["1"])
        self.assertEqual(len(metrics["pooled"]["confusion_matrix"]), 2)
        self.assertTrue((output / f"{model}_predictions.csv").exists())
        predictions = pd.read_csv(output / f"{model}_predictions.csv")
        self.assertTrue(predictions["slice_file_name"].is_unique)
        self.assertTrue((predictions["fold"] == 1).all())
        extension = "pt" if model == "cnn" else "joblib"
        self.assertTrue((output / f"{model}_fold1.{extension}").exists())
        return metrics

    def test_svm_synthetic_dataset_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            report = self.run_cli(root, "svm")
            self.assertIn("accuracy", report["fold_mean_std"])
            self.assertIn("macro_f1", report["pooled"])

    def test_cnn_synthetic_dataset_cli(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("CNN smoke test requires torch from requirements.txt")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            report = self.run_cli(root, "cnn")
            self.assertEqual(report["config"]["seed"], 122333)
            self.assertTrue(report["folds"]["1"]["selected_epochs"] >= 1)

    def test_label_order_is_shared_by_metrics_and_confusion_matrix(self):
        result = baselines.metrics(np.array(["a", "b", "a"]),
                                   np.array(["b", "b", "a"]), ["b", "a"])
        self.assertEqual(list(result["per_class"]), ["b", "a"])
        self.assertEqual(result["confusion_matrix"], [[1, 0], [1, 1]])

    def test_svm_pipeline_fits_only_non_test_fold_rows(self):
        X = np.arange(12, dtype=np.float32).reshape(6, 2)
        y = np.array(["a", "b", "a", "b", "a", "b"])
        folds = np.array([1, 1, 2, 2, 3, 3])
        captured = {}
        class SpyPipeline:
            def fit(self, x, target):
                captured["X"] = x.copy()
                captured["y"] = target.copy()
            def predict(self, x):
                return np.array(["a", "b"])
        def make_spy(*steps):
            captured["steps"] = steps
            return SpyPipeline()
        with mock.patch.object(baselines, "make_pipeline", side_effect=make_spy), \
             mock.patch.object(baselines, "yield_artifact"):
            baselines.evaluate_arrays(X, y, folds, ["a", "b"], selected_folds=[2])
        np.testing.assert_array_equal(captured["X"], X[folds != 2])
        np.testing.assert_array_equal(captured["y"], y[folds != 2])
        self.assertIsInstance(captured["steps"][0], baselines.StandardScaler)

    def test_logmel_silence_and_short_clip_have_fixed_shape_and_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            silence = Path(temporary) / "silence.wav"
            short = Path(temporary) / "short.wav"
            sf.write(silence, np.zeros(100, dtype=np.float32), 22050)
            sf.write(short, np.ones(1000, dtype=np.float32) * .1, 22050)
            silence_feature = baselines.logmel_image(silence)
            short_feature = baselines.logmel_image(short)
        self.assertEqual(silence_feature.shape, (64, 173))
        self.assertTrue(np.all(silence_feature == 0.))
        self.assertEqual(short_feature.shape, (64, 173))
        self.assertGreaterEqual(float(short_feature.min()), 0.)
        self.assertLessEqual(float(short_feature.max()), 1.)

    def test_cache_reused_then_invalidated_by_source_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio" / "fold1"
            metadata = root / "metadata"
            audio.mkdir(parents=True)
            metadata.mkdir()
            file = audio / "a.wav"
            file.write_bytes(b"one")
            pd.DataFrame([{"fold": 1, "slice_file_name": "a.wav", "class": "a", "classID": 0}]
                         ).to_csv(metadata / "UrbanSound8K.csv", index=False)
            cache = root / "mfcc.npz"
            with mock.patch.object(baselines, "mfcc_vector", return_value=np.ones(80, np.float32)) as extract:
                baselines.load_or_create_data(root, "svm", cache)
                baselines.load_or_create_data(root, "svm", cache)
                self.assertEqual(extract.call_count, 1)
                file.write_bytes(b"changed source")
                baselines.load_or_create_data(root, "svm", cache)
                self.assertEqual(extract.call_count, 2)
                with mock.patch.dict(baselines.MFCC_CONFIG, {"n_mfcc": 41}):
                    baselines.load_or_create_data(root, "svm", cache)
                self.assertEqual(extract.call_count, 3)

    def test_cnn_fit_uses_cyclic_validation_then_all_non_test_folds(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CNN protocol test requires torch")
        folds = np.repeat(np.arange(1, 11), 2)
        X = np.arange(20, dtype=np.float32).reshape(-1, 1)
        y = np.tile(np.array(["a", "b"]), 10)
        fake_model = mock.Mock()
        fake_model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
        calls = []
        def fake_fit(fit_x, fit_y, labels, seed, epochs, batch_size, validation=None):
            calls.append((fit_x.copy(), None if validation is None else validation[0].copy()))
            return fake_model, 3
        with mock.patch.object(baselines, "_fit_cnn", side_effect=fake_fit), \
             mock.patch.object(baselines, "_predict_batched", return_value=np.zeros(2, dtype=int)):
            pred, _, selected = baselines._evaluate_cnn_fold(
                X, y, folds, 1, ["a", "b"], 122333, 30, 64)
        np.testing.assert_array_equal(calls[0][0].ravel(), X[(folds != 1) & (folds != 2)].ravel())
        np.testing.assert_array_equal(calls[0][1].ravel(), X[folds == 2].ravel())
        np.testing.assert_array_equal(calls[1][0].ravel(), X[folds != 1].ravel())
        self.assertEqual(selected, 3)
        self.assertEqual(len(pred), 2)

    def test_cnn_prediction_batches_are_bounded(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CNN batch test requires torch")
        class CountingModel:
            def __init__(self):
                self.sizes = []
            def __call__(self, batch):
                self.sizes.append(len(batch))
                return torch.zeros((len(batch), 2))
        model = CountingModel()
        result = baselines._predict_batched(model, np.zeros((5, 64, 173), np.float32),
                                             torch.device("cpu"), batch_size=2)
        self.assertEqual(model.sizes, [2, 2, 1])
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
