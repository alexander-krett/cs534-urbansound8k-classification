"""Fast AFM contract tests; model execution is mocked and needs no weights."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import afm_features as afm


class AFMFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.cache = self.root / "cache"
        self.checkpoints = self.root / "checkpoints"
        self.metadata = self.data / "metadata"
        self.metadata.mkdir(parents=True)
        (self.data / "audio" / "fold1").mkdir(parents=True)
        self.rows = [
            {"slice_file_name": f"sample{i}.wav", "fold": 1,
             "class": name, "classID": i}
            for i, name in enumerate(afm.CLASSES[:2])
        ]
        for row in self.rows:
            (self.data / "audio" / f"fold{row['fold']}" / row["slice_file_name"]).write_bytes(b"mock audio")
        pd.DataFrame(self.rows).to_csv(self.metadata / "UrbanSound8K.csv", index=False)
        self.checkpoints.mkdir()
        (self.checkpoints / afm.CLAP_CHECKPOINT).write_bytes(b"mock clap")
        (self.checkpoints / afm.BEATS_CHECKPOINT).write_bytes(b"mock beats")
        beats_src = self.checkpoints / afm.BEATS_SOURCE_DIR
        beats_src.mkdir()
        (beats_src / "REVISION").write_text(afm.BEATS_SOURCE_REVISION)
        for source_file in ("BEATs.py", "backbone.py", "modules.py"):
            (beats_src / source_file).write_text("# mock source\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_metadata_schema_and_official_class_mapping(self):
        frame = afm.load_metadata(self.data)
        self.assertEqual(list(frame.columns), ["sample_id", "slice_file_name", "fold", "class", "classID"])
        self.assertEqual(frame["sample_id"].tolist(), ["fold1/sample0.wav", "fold1/sample1.wav"])
        self.assertEqual(afm.CLASSES[0], "air_conditioner")
        self.assertEqual(len(afm.PROMPTS), 10)
        self.assertTrue(all(prompt.startswith("An audio recording of ") and prompt.endswith(".")
                            for prompt in afm.PROMPTS))

    def test_invalid_class_id_mapping_is_rejected(self):
        bad = pd.DataFrame([{"slice_file_name": "sample0.wav", "fold": 1,
                             "class": "car_horn", "classID": 0}])
        bad.to_csv(self.metadata / "UrbanSound8K.csv", index=False)
        with self.assertRaisesRegex(ValueError, "official UrbanSound8K mapping"):
            afm.load_metadata(self.data)

    def test_l2_normalization_and_independent_fusion(self):
        clap = np.array([[3., 4.], [0., 0.]], dtype=np.float32)
        beats = np.array([[0., 2., 0.], [1., 0., 0.]], dtype=np.float32)
        fused = afm.fuse_embeddings(clap, beats)
        np.testing.assert_allclose(fused[0], [0.6, 0.8, 0., 1., 0.])
        np.testing.assert_allclose(fused[1], [0., 0., 1., 0., 0.])

    def test_repeatpad_matches_official_short_clip_policy_and_long_truncation(self):
        with mock.patch.object(afm, "CLAP_SAMPLES", 7), \
             mock.patch.object(afm, "_audio", wraps=afm._audio), \
             mock.patch("librosa.load", return_value=(np.array([1., 2., 3.], np.float32), 48_000)):
            actual = afm._audio("mock.wav", 48_000, 7)
        np.testing.assert_array_equal(actual, [1., 2., 3., 1., 2., 3., 0.])

    def test_short_beats_input_is_padded_and_invalid_frames_excluded(self):
        import torch
        class Clap:
            def get_audio_embedding_from_data(self, audio, use_tensor=False):
                self.audio = audio
                return np.ones((1, 512), np.float32)
        class Beats:
            def extract_features(self, audio, padding_mask):
                self.audio, self.mask = audio, padding_mask
                features = torch.stack((torch.ones(768), torch.full((768,), 99.)))[None]
                return features, torch.tensor([[False, True]])
        clap, beats = Clap(), Beats()
        with mock.patch.object(afm, "_audio", side_effect=[np.ones(480000, np.float32), np.ones(800, np.float32)]):
            c, b = afm._embed_audio(clap, beats, ["mock.wav"], "cpu")
        self.assertEqual(beats.audio.shape, (1, 3200))
        self.assertEqual(int(beats.mask.sum()), 2400)
        np.testing.assert_array_equal(b, np.ones((1, 768)))
        self.assertEqual(c.shape, (1, 512))

    def test_corrupt_and_misaligned_cache_is_rejected(self):
        frame = afm.load_metadata(self.data)
        path = self.cache / "afm_features.npz"
        self.cache.mkdir()
        path.write_bytes(b"not a numpy archive")
        self.assertIsNone(afm._read_cache(path, {}, frame))
        afm._atomic_save(path, frame.iloc[::-1], np.ones((2, 512)), np.ones((2, 768)), {})
        self.assertIsNone(afm._read_cache(path, {}, frame))

    def test_cache_roundtrip_reuse_and_extract_disabled(self):
        frame = afm.load_metadata(self.data)
        signature = {"token": "stable", "checkpoints": {}}
        clap = np.ones((2, afm.CLAP_DIM), np.float32)
        beats = np.ones((2, afm.BEATS_DIM), np.float32)
        path = self.cache / "afm_features.npz"
        afm._atomic_save(path, frame, clap, beats, signature)
        loaded = afm._read_cache(path, signature, frame)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded[0], clap)
        self.assertIsNone(afm._read_cache(path, {"token": "stale"}, frame))
        with mock.patch.object(afm, "_signature", return_value=signature):
            result_frame, features, provenance = afm.load_or_extract(
                self.data, self.cache, self.checkpoints, device="cpu", allow_extract=False)
        self.assertEqual(result_frame.sample_id.tolist(), frame.sample_id.tolist())
        self.assertEqual(features["clap"].shape, (2, afm.CLAP_DIM))
        self.assertEqual(features["beats"].shape, (2, afm.BEATS_DIM))
        self.assertTrue(provenance["cache_hit"])
        with mock.patch.object(afm, "_signature", return_value={"token": "missing"}):
            with self.assertRaisesRegex(RuntimeError, "extraction is disabled"):
                afm.load_or_extract(self.data, self.cache, self.checkpoints,
                                    device="cpu", allow_extract=False)

    def test_mocked_extraction_writes_aligned_cache(self):
        frame = afm.load_metadata(self.data)
        signature = {"signature": "mock", "checkpoints": {}}
        clap = np.full((2, afm.CLAP_DIM), 2, np.float32)
        beats = np.full((2, afm.BEATS_DIM), 3, np.float32)
        with mock.patch.object(afm, "_signature", return_value=signature), \
             mock.patch.object(afm, "_load_models", return_value=(object(), object())), \
             mock.patch.object(afm, "_embed_audio", return_value=(clap, beats)) as extract:
            result_frame, features, provenance = afm.load_or_extract(
                self.data, self.cache, self.checkpoints, device="cpu")
            self.assertEqual(extract.call_args.args[2], [
                str(self.data / "audio" / "fold1" / "sample0.wav"),
                str(self.data / "audio" / "fold1" / "sample1.wav")])
        self.assertEqual(len(result_frame), 2)
        np.testing.assert_array_equal(features["clap"], clap)
        np.testing.assert_array_equal(features["beats"], beats)
        self.assertFalse(provenance["cache_hit"])
        with np.load(self.cache / "afm_features.npz", allow_pickle=False) as data:
            self.assertEqual(json.loads(str(data["signature"].item())), signature)


if __name__ == "__main__":
    unittest.main()
