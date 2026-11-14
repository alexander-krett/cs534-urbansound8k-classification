# AFM report artifacts

Snapshot from successful WPI Turing Slurm job **2517905** on November 13, 2026
(exit code 0, elapsed 4m 58s). This job reused the complete embedding cache
produced by job 2517889. The earlier job completed extraction but failed while
collecting Git metadata; the accompanying source/test fix makes that optional.

- `afm_results.md` and `.tex`: comparison tables, using the mean and population
  standard deviation across the ten official UrbanSound8K folds.
- `*_metrics.json`: all fold metrics, pooled metrics, per-class metrics,
  confusion matrices, model settings, prompts, checkpoint hashes, package
  versions, and source hashes.
- `*_predictions.csv`: one held-out prediction per clip, 8,732 unique clips for
  each of seven evaluations. These support paired comparisons and error analysis.
- `*_confusion_matrix.png`: pooled confusion matrices for report figures.
- `afm_manifest.json`: shared run provenance and baseline availability.

Best fold-mean accuracy: CLAP+BEATs fusion with SVM, **89.6861%**. Its pooled
accuracy is **89.7160%**; distinguish these aggregation methods in the report.
CLAP uses one fixed prompt per class, with no prompt ensembling or tuning.

The classical baseline comparison rows are unavailable because their metrics
manifests were absent. Existing SVM/CNN predictions remain committed under
`models/` and were not replaced by this snapshot.

The run emitted CuBLAS nondeterminism warnings during MLP training; fixed seeds
do not guarantee identical MLP results on reruns. These are measured scores,
not a claim of bitwise reproducibility or statistically significant gains.

Training checkpoints, downloaded foundation models, embedding caches, and full
Slurm logs are excluded. Raw outputs remain locally under `models/afm/`;
this snapshot is kept separately so future runs do not overwrite it.
