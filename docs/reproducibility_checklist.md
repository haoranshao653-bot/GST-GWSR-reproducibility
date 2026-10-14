# Reproducibility checklist

- [ ] Confirm licenses and versions for the IHFC source and all predictor products.
- [ ] Create `data/processed/heat_flow_stations.csv` with the required schema.
- [ ] Run `scripts/validate_station_table.py` and archive its JSON report.
- [ ] Record Python and package versions with `scripts/check_environment.py`.
- [ ] Run training-only `k` selection and verify `selected_k=45`.
- [ ] Run training-only `alpha–tau` selection and verify `(0.25, 0.7)`.
- [ ] Run the three-seed outer comparison with the locked configuration.
- [ ] Compare metrics, prediction files, formula files, and protection counts with `results/final_outer/`.
- [ ] Record hardware and wall-clock time; runtime is not expected to match across machines.
- [ ] Keep the outer evaluation responses out of every tuning command.
