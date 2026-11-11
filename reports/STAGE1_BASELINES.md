# Stage 1 model comparisons and baselines

## Comparable evaluation (same 3-fold / seed 42)

| Model | PR-AUC | ROC-AUC | Brier ↓ | Top10 hits |
|---|---:|---:|---:|---:|
| N16 | 0.355313 | 0.725714 | 0.091568 | 6 |
| N17 + Friend LR | 0.355652 | 0.731860 | 0.091356 | 6 |
| **N17 + Friend Hybrid** | **0.357247** | **0.734447** | **0.091106** | **6** |

This aligned evaluation provides an ablation of the extra friend model: LR vs the full LR/quantum-kernel/VQC hybrid. It is a *single-seed exploratory comparison* and the gains are small; it does not demonstrate generalization superiority or statistical significance. Underlying saved summary: `results/stage1_cv/n17_aligned_3fold_summary.csv`.

## Historical exploratory results (different protocol, not directly comparable)

| Model | PR-AUC | ROC-AUC | Mean top10 hits |
|---|---:|---:|---:|
| F2 only | 0.3470 | 0.7062 | 5.67 |
| F2 70% + Friend LR 30% | ~0.3606 | ~0.7110 | 6.33 |

These are from earlier experiments with different evaluation protocols. They must **not** be treated as a paired test of N17. Values transcribed from the development record; not all original full Fold-level runs are included in this archive.

## Reproducibility notes

- Main submission model: N17 + Friend Hybrid, fixed history gamma = 0.2, fitted on all 300 training cases for the 100-person test.
- `code/atomiq_N17_same_nested_three_models.py` and dependent scripts permit fresh nested-CV evaluation but require the original external `qcampaign` package from the teammate's repository.
- The teammate repository snapshot is **bundled** at `code/friend_atomiq/`. Its original cross-validation CSVs are at `code/friend_atomiq/results/`; its independently selected final campaign differs from the primary N17 result. See `README.md` and `code/friend_SOURCE_PROVENANCE.md`.
- No test-label tuning was carried out, but the 300-row development data was used repeatedly for model exploration.
- The original Qiskit model and the numpy baseline are not identical evaluation procedures.

## Teammate original independent baselines

The teammate also provides `code/friend_atomiq/results/stage1_cv_metrics.csv`, `stage1_cv_per_repeat.csv`, and `stage1_oof_predictions_train.csv`. These results were obtained with the teammate's **different nested 5×5 CV protocol**, and should not be pooled with N17's 3-fold single-seed comparisons as though they were the same benchmark.
