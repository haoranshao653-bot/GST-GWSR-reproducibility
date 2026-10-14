# GST–GWSR reproducibility repository

This repository contains the reproducibility package for the GST–GWSR marine heat-flow study: source data, the derived station table, the manuscript model code, locked configurations, figures, and recorded outputs. It contains only the final protocol used for the reported results.

The canonical protocol is the protected fixed outer split:

- 1,024 marine heat-flow stations and eight geological/geophysical predictors.
- An 80:20 split with seed 42: 819 outer-training rows and 205 held-out rows.
- Training-only 3-IQR cleaning retains 815 rows.
- Predictor standardization and log-response standardization are fitted on the retained outer training rows only.
- Adaptive bisquare kernel with `k=45`, selected by five-fold training-only cross-validation.
- GST uses distance similarity and local predictor-distribution similarity. The selected grid setting is `alpha=0.25`, `tau=0.7`.
- GP seeds are 42, 123, and 456; population size is 1,000; complete search is 20 generations; transfer adaptation is 5 generations.
- Non-finite or unsafe standardized predictions use the predefined local adaptive-bisquare weighted-mean fallback.

The final recorded three-seed outer ensemble is:

| Model | R² | RMSE (mW m⁻²) | MAE (mW m⁻²) |
|---|---:|---:|---:|
| Independent GWSR | 0.6341 | 14.0861 | 10.0628 |
| GST-GWSR | 0.6470 | 13.8344 | 9.9883 |

The matched `k=45` efficiency comparison records a mean runtime reduction from 93.86 s to 60.99 s (35.03%) and a reduction in objective-function evaluations from 4.100 million to 1.608 million (60.78%). These values are hardware- and implementation-dependent.

## Repository map

```text
configs/                 Locked protocol descriptions
src/gwsr/                Portable core and exact experiment adapters
scripts/                 Data validation, run wrappers, and figure helpers
data/raw/                Source products, source metadata, and predictor grids
data/processed/          Exact derived station table used by the experiments
data/example/            Small deterministic smoke-test input
results/                 Recorded metrics, predictions, manifests, and audits
figures/current/         Current paper-ready figures
docs/                    Data provenance and reproducibility protocol
tests/                   Lightweight structural checks
```

## Data and Git LFS

The exact model input is `data/processed/heat_flow_stations.csv` with columns `lon`, `lat`, `heat_flow`, `topo`, `bouguer`, `sediment`, `moho`, `lab`, `cpd`, `lithos_km`, and `volcano`. Large source rasters are tracked with Git LFS, so a clone must run `git lfs pull`.

To reproduce the paper run, obtain the permitted source products, create the derived station table, and validate it:

```powershell
python scripts/validate_station_table.py --input data/processed/heat_flow_stations.csv
```

The validation script checks column names, finite values, positive response values, row count, and the expected eight-predictor schema. It does not download data or silently alter the table.

## Environment

The recorded run used Python 3.11.17 on Windows. Create an isolated environment and install the pinned scientific dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The exact adapters use private `gplearn` operator APIs and therefore require `gplearn==0.4.3`. Run `python scripts/check_environment.py` before a long run.

Before using the full station table, verify the software path with the bounded smoke test:

```powershell
python scripts/run_smoke.py
```

## Reproduction order

1. Prepare and validate `data/processed/heat_flow_stations.csv`.
2. Run the training-only neighbourhood selection with the exact random five-fold adapter. The outer-test responses are not used during selection:

   ```powershell
   python src/gwsr/exact/run_random5fold.py
   ```

3. Run the training-only `alpha–tau` grid at fixed `k=45`:

   ```powershell
   $env:GST_GWSR_STATION_FILE = (Resolve-Path data/processed/heat_flow_stations.csv)
   python src/gwsr/exact/select_alpha_tau_exact.py
   ```

4. Run the locked outer comparison. Set `GST_GWSR_STATION_FILE` and run `src/gwsr/exact/run_outer_gwsr.py`, then run `src/gwsr/exact/run_outer_gst.py`. The scripts read the selected training-only alpha–tau record and write fresh outputs under `run_outputs/`.
5. Generate figures and compare the new manifests with the recorded outputs in `results/`.

The recorded results are evidence of one fixed study region and station configuration. They are not a claim of cross-region generalization.

## Results and provenance

`results/final_outer/` contains the primary three-seed outer comparison. `results/gwsr_cv_k45/` contains the exact training-only selection records and fold-level outputs. `results/supplementary_efficiency/` contains the descriptive efficiency analysis. Every reported result is accompanied by its configuration and code fingerprint.

## License and citation

Code is released under the MIT license in `LICENSE`. Data remain subject to their original providers' terms; see `data/README.md`. If the study is used, cite the paper and the source datasets listed in `docs/data_sources.md`.
