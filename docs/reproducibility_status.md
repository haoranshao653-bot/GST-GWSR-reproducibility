# Reproducibility status at packaging time

The recorded experiment environment is Python 3.11.17 on Windows. The locked environment is NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, PyYAML 6.0.3, and gplearn 0.4.3.

Completed checks:

- Python source compilation passed for `src/`, `scripts/`, and `tests/`.
- The station-table validator passed on the deterministic 32-row smoke table.
- A bounded smoke test for the final code path completed and wrote predictions, a manifest, and strategy counts.
- The exact adapters import and compile successfully after replacing desktop paths with repository-relative or environment-controlled paths.
- The recorded primary metrics match `results/final_outer/outer_ensemble_metrics.csv` to the shown precision.

The full 1,024-station rerun was not started during packaging because a complete GP search is computationally expensive. The repository includes the exact derived table and recorded outputs needed to inspect the reported run; the short smoke test is a software-path check and is not a replacement for the full experiment. The exact input table SHA-256 matches `data/processed/manifest.json`.
