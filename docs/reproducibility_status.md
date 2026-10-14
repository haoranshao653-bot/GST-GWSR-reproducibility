# Reproducibility status at packaging time

The public copy was checked on Windows with Python 3.14 in an isolated virtual environment. The installed verification environment reported NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, and gplearn 0.4.3.

Completed checks:

- Python source compilation passed for `src/`, `scripts/`, and `tests/`.
- The station-table validator passed on the deterministic 32-row smoke table.
- The bounded end-to-end GST smoke test completed and wrote predictions, a manifest, and strategy counts.
- The exact adapters import and compile successfully after replacing desktop paths with repository-relative or environment-controlled paths.
- The recorded primary metrics match `results/final_outer/outer_ensemble_metrics.csv` to the shown precision.

The full 1,024-station rerun was not started during packaging because a complete GP search is computationally expensive. The private advisor version includes the derived table, source products, and processing code. The exact input table SHA-256 matches `data/processed/manifest.json`.
