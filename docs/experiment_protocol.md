# Experiment protocol

The outer test is fixed before model fitting: `train_test_split(..., test_size=0.20, random_state=42)`. The 819-row outer training partition is cleaned with a 3-IQR rule on `heat_flow`; four rows are removed, leaving 815 rows. All scalers and the log-response transformation are fitted inside the training side of each inner fold or the final outer-training partition.

Neighbourhood size is selected from the declared grid using five-fold training-only validation and the three GP seeds. The selected value is `k=45`. With `k` fixed, the GST grid is `alpha ∈ {0, 0.25, 0.5, 0.75, 1}` and `tau ∈ {0.5, 0.7, 0.9}`. The selected pair is `(0.25, 0.7)` by mean training-only CV RMSE. The 205 outer-evaluation responses are read only after these choices are locked.

Independent GWSR and GST-GWSR use the same local objective, adaptive bisquare weights, GP population, generations, seeds, and prediction guard. GST transfers populations only between eligible local tasks; complete search is retained as a fallback. The reported ensemble is the pointwise arithmetic mean of the three seed predictions.

The cross-`k` efficiency analysis is descriptive. It does not re-select `k` or GST parameters from the outer responses.
