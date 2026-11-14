"""Lightweight scheduler safety and artifact setup tests."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('run_afm', Path(__file__).resolve().parents[1] / 'src/run_afm.py')
af = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(af)


class SchedulerSafetyTests(unittest.TestCase):
    def test_production_stops_before_loading_encoders_on_login(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Slurm allocation'):
                af.main(['--stage', 'all'])

    def test_setup_stops_before_network_on_login(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(af.urllib.request, 'urlretrieve') as retrieve:
            with self.assertRaisesRegex(RuntimeError, 'Slurm allocation'):
                af.setup(Path('/unused'), 'https://unused')
            retrieve.assert_not_called()

    def test_rejects_html_download_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.pt'
            def fake_download(url, path):
                Path(path).write_bytes(b'<!DOCTYPE html><html>sign in</html>')
            with patch.object(af.urllib.request, 'urlretrieve', side_effect=fake_download):
                with self.assertRaisesRegex(RuntimeError, 'expected artifact'):
                    af.download('https://example.invalid', target)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_allocation_environment_on_login_host_is_rejected(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_NODELIST': 'gpu01'}, clear=True), \
             patch.object(af.subprocess, 'run', return_value=af.subprocess.CompletedProcess([], 0, 'gpu01\n', '')), \
             patch.object(af.socket, 'gethostname', return_value='login-03'):
            with self.assertRaisesRegex(RuntimeError, 'outside the Slurm allocation'):
                af.require_slurm()

    def test_allocated_compute_host_is_accepted(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_NODELIST': 'gpu01'}, clear=True), \
             patch.object(af.subprocess, 'run', return_value=af.subprocess.CompletedProcess([], 0, 'gpu01\n', '')), \
             patch.object(af.socket, 'gethostname', return_value='gpu01.cluster'):
            af.require_slurm()

    def test_thread_settings_follow_allocation(self):
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '2'}, clear=True):
            af.configure_threads()
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '2')


if __name__ == '__main__':
    unittest.main()
