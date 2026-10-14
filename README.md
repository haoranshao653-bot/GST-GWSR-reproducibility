# GST–GWSR reproducibility repository

This repository contains the complete private advisor package for the GST–GWSR marine heat-flow study: source data, processing scripts, the exact derived station table, model code, configurations, figures, and recorded outputs. It is organized for inspection and rerunning rather than for distributing a private desktop workspace.

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

The exact model input is `data/processed/heat_flow_stations.csv` with columns `lon`, `lat`, `heat_flow`, `topo`, `bouguer`, `sediment`, `moho`, `lab`, `cpd`, `lithos_km`, and `volcano`. The private advisor version includes this table, the raw/source products, and the processing scripts. Large source rasters are tracked with Git LFS, so a clone must run `git lfs pull`.

To reproduce the paper run, obtain the permitted source products, create the derived station table, and validate it:

```powershell
python scripts/validate_station_table.py --input data/processed/heat_flow_stations.csv
```

The validation script checks column names, finite values, positive response values, row count, and the expected eight-predictor schema. It does not download data or silently alter the table.

## Environment

Python 3.11–3.14 is supported by the recorded run. Create an isolated environment and install the pinned scientific dependencies:

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
2. Run the training-only neighbourhood selection. The exact adapter writes to `run_outputs/k_selection` and never uses outer-test responses:

   ```powershell
   python src/gwsr/select_k_exact.py --direct-confirm --direct-k 20 25 30 35 40 45 50 55 60
   ```

3. Run the training-only `alpha–tau` grid at fixed `k=45`:

   ```powershell
   $env:GST_GWSR_STATION_FILE = (Resolve-Path data/processed/heat_flow_stations.csv)
   python src/gwsr/exact/select_alpha_tau_exact.py
   ```

4. Run the locked outer comparison. Set `GST_GWSR_STATION_FILE` and run `src/gwsr/exact/run_outer_gwsr.py`, then run `src/gwsr/exact/run_outer_gst.py`. The scripts read the selected training-only alpha–tau record and write fresh outputs under `run_outputs/`.
5. Generate figures and compare the new manifests with the recorded outputs in `results/`.

The recorded results are evidence of one fixed study region and station configuration. They are not a claim of cross-region generalization.

## Sharing with the advisor before submission

Create a private GitHub repository from this directory, add the advisor as a collaborator, and send the repository URL. The private repository is the working review copy; keep the `main` branch at the code and data version used in the manuscript. Before making the repository public after submission or acceptance, review the complete Git history and create a versioned GitHub Release. The publication workflow is described in `docs/publication_workflow.md`.

## Results and provenance

`results/final_outer/` contains the primary three-seed outer comparison. `results/gwsr_cv_k45/` and `results/k_selection_audit/` contain the training-only selection records and fold-level outputs. `results/supplementary_efficiency/` contains the descriptive cross-`k` efficiency analysis. `results/baselines/validation_audit/` contains the baseline-protocol audit outputs. Every public result should be accompanied by its configuration and code fingerprint; absolute desktop paths in legacy records are retained only in the archived evidence and are not used by the portable adapters.

## License and citation

Code is released under the MIT license in `LICENSE`. Data remain subject to their original providers' terms; see `data/README.md`. If the study is used, cite the paper and the source datasets listed in `docs/data_sources.md`.
