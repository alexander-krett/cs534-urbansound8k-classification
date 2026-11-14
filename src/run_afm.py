"""Slurm-only entry point for frozen CLAP/BEATs experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import importlib.metadata
import platform
import socket
from pathlib import Path
import tempfile
import urllib.request

UNILM_REVISION = '31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd'
CLAP_REVISION = 'b3708341862f581175dba5c356a4ebf74a9b6651'
BEATS_URL = 'https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?download=1'


def require_slurm() -> None:
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_NODELIST'):
        raise RuntimeError('AFM production work requires a Slurm allocation. Submit cluster/run_afm.sbatch; only --synthetic-smoke is permitted locally.')
    # salloc may leave its shell on the login host. Check the actual execution
    # host, rather than accepting allocation variables as proof of compute routing.
    nodes = subprocess.run(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
                           capture_output=True, text=True, check=True)
    allocated = {host.split('.')[0] for host in nodes.stdout.split()}
    if socket.gethostname().split('.')[0] not in allocated:
        raise RuntimeError('This host is outside the Slurm allocation. Execute through sbatch or srun on an allocated compute node.')


def configure_threads() -> None:
    count = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(count)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('MPLBACKEND', 'Agg')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, minimum_size: int = 1) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        try:
            urllib.request.urlretrieve(url, temporary)
        except OSError as error:
            raise RuntimeError(f"Could not download {target.name}: {error}. If this is BEATs, pre-stage the official pretrained iter3+ AS2M checkpoint at {target} or set BEATS_CHECKPOINT_URL to a direct download URL.") from error
        with temporary.open('rb') as handle:
            prefix = handle.read(256).lstrip().lower()
        if temporary.stat().st_size < minimum_size or prefix.startswith((b'<!doctype html', b'<html')):
            raise RuntimeError(f'Download did not return the expected artifact: {url}. For BEATs, obtain the pretrained iter3+ AS2M checkpoint from Microsoft and place it at {target}, or pass --beats-url with a direct download URL.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def setup(checkpoint_dir: Path, beats_url: str) -> None:
    require_slurm()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        '630k-audioset-best.pt': f'https://huggingface.co/lukewys/laion_clap/resolve/{CLAP_REVISION}/630k-audioset-best.pt',
        'BEATs_iter3_plus_AS2M.pt': beats_url,
    }
    for name, url in weights.items():
        path = checkpoint_dir / name
        if not path.exists():
            print(f'Downloading {name}', flush=True)
            download(url, path, minimum_size=1_000_000)
        if path.stat().st_size < 1_000_000:
            raise ValueError(f'Checkpoint appears incomplete: {path}')
    source = checkpoint_dir / 'unilm-beats'
    source.mkdir(exist_ok=True)
    for name in ('BEATs.py', 'backbone.py', 'modules.py'):
        # Small pinned sources are refreshed to prevent mixed revisions.
        download(f'https://raw.githubusercontent.com/microsoft/unilm/{UNILM_REVISION}/beats/{name}', source / name)
    download(f'https://raw.githubusercontent.com/microsoft/unilm/{UNILM_REVISION}/LICENSE', source / 'LICENSE')
    (source / 'REVISION').write_text(UNILM_REVISION + '\n')
    manifest = {'clap_revision': CLAP_REVISION, 'beats_source_revision': UNILM_REVISION,
                'artifacts': {str(p.relative_to(checkpoint_dir)): sha256(p)
                              for p in [*(checkpoint_dir / name for name in weights), *(source / name for name in ('BEATs.py', 'backbone.py', 'modules.py'))]}}
    (checkpoint_dir / 'setup_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Checkpoint setup complete.', flush=True)


def git_output(repository: Path, *arguments: str) -> str:
    """Git provenance is optional on compute nodes; source hashes remain recorded."""
    fallback = os.environ.get('AFM_GIT_REVISION', 'unknown') if arguments == ('rev-parse', 'HEAD') else 'unknown'
    try:
        result = subprocess.run(['git', *arguments], cwd=repository, capture_output=True,
                                text=True, check=False)
    except OSError:
        return fallback
    return result.stdout.strip() if result.returncode == 0 else fallback


def synthetic_smoke() -> None:
    import numpy as np
    from afm_features import fuse_embeddings, l2_normalize, PROMPTS
    rng = np.random.default_rng(122333)
    clap, beats = rng.normal(size=(3, 512)), rng.normal(size=(3, 768))
    fused = fuse_embeddings(clap, beats)
    assert fused.shape == (3, 1280)
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(clap), axis=1), 1)
    assert len(PROMPTS) == 10
    print('Synthetic AFM smoke check passed; no checkpoints loaded.')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('setup', 'extract', 'evaluate', 'all', 'smoke'), default='all')
    parser.add_argument('--synthetic-smoke', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=Path('data/UrbanSound8K'))
    parser.add_argument('--cache-dir', type=Path, default=Path('features/afm'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/afm'))
    parser.add_argument('--output-dir', type=Path, default=Path('models/afm'))
    parser.add_argument('--baseline-dir', type=Path, default=Path('models'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seed', type=int, default=122333)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--beats-url', default=os.environ.get('BEATS_CHECKPOINT_URL', BEATS_URL))
    args = parser.parse_args(argv)
    configure_threads()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    require_slurm()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')
    if args.stage == 'setup':
        setup(args.checkpoint_dir, args.beats_url)
        return
    from afm_features import load_or_extract, text_prototypes
    # Smoke caches are isolated and cannot be mistaken for full dataset caches.
    cache_dir = args.cache_dir / 'smoke' if args.stage == 'smoke' else args.cache_dir
    frame, embeddings, provenance = load_or_extract(args.data_dir, cache_dir, args.checkpoint_dir,
                                                  device=args.device, smoke_limit=10 if args.stage == 'smoke' else None,
                                                  allow_extract=args.stage != 'evaluate')
    if args.stage == 'extract':
        print(f'Embedding caches ready: {len(frame)} samples.')
        return
    prototypes, text_provenance = text_prototypes(args.checkpoint_dir, device=args.device)
    provenance['text'] = text_provenance
    repository = Path(__file__).resolve().parents[1]
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'pandas', 'scikit-learn', 'torch', 'torchaudio', 'laion-clap', 'librosa'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    provenance['run'] = {
        'git_revision': git_output(repository, 'rev-parse', 'HEAD'),
        'working_tree_status': git_output(repository, 'status', '--porcelain'),
        'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*afm*.py')},
        'versions': versions, 'seed': args.seed, 'epochs_max': args.epochs,
        'batch_size': args.batch_size, 'device': args.device,
        'svm': {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'},
        'mlp': {'hidden_dim': 256, 'dropout': .3, 'optimizer': 'Adam',
                'learning_rate': .001, 'patience': 10,
                'validation': 'next official fold cyclically; fresh scaler/model refit all nine training folds'},
    }
    if args.stage == 'smoke':
        import numpy as np
        assert prototypes.shape == (10, 512) and np.isfinite(prototypes).all()
        print(f'Real-encoder Slurm smoke passed for {len(frame)} samples; no probes trained.')
        return
    from afm_evaluate import evaluate_all
    evaluate_all(frame, embeddings, prototypes, args.output_dir, provenance,
                 seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                 baseline_dir=args.baseline_dir)


if __name__ == '__main__':
    main()
