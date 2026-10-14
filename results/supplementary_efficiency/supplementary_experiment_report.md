# Supplementary GST-GWSR experiments

## Protocol

The fixed outer split, three GP seeds (42, 123, 456), adaptive-bisquare kernel, population size 1000, 20-generation complete search, five-generation GST adaptation, and numerical protection were retained. The final GST association parameters were fixed at alpha = 0.25 and tau = 0.7. Results across k are descriptive sensitivity results and were not used to re-select k or any GST parameter.

## Completed experiments

- Paired outer-test comparison for k = 20, 25, 30, 35, 40, 45, 50, 55, and 60.
- Three matched seeds for both independent GWSR and GST-GWSR at every k.
- Runtime, objective-function evaluations, R², RMSE, MAE, transfer-task counts, complete-search counts, and protection counts recorded.
- Training-only alpha–tau runtime/accuracy trade-off plotted from the completed 15-configuration CV grid.

## Interpretation

GST reduced mean wall-clock time at every tested k, with reductions from 32.55% to 34.21%. Objective-function evaluations decreased by 60.10%–60.78% across the grid. This supports a robust search-cost reduction under the fixed implementation.
Predictive changes were not uniformly positive across k. Mean RMSE reductions were -3.82%–4.28%; positive mean reductions occurred at k = 20, 25, 30, 45, 50, 60, whereas k = 35, 40, 55 showed no mean improvement. The paper should therefore claim stable efficiency reduction, not universal accuracy improvement.
At the selected k = 45, the fixed-parameter rerun gave a mean RMSE reduction of 1.64% and a mean runtime reduction of 33.36%. The original final three-seed ensemble remains the primary accuracy result; this supplementary grid is a robustness analysis.
The selected alpha–tau configuration had mean CV RMSE 20.7568 and mean CV runtime 353.37 s. Runtime was not used to override the minimum-RMSE selection.

## Files

- `outer_seed_detail.csv`: all paired seed/model results.
- `outer_seed_paired_efficiency.csv`: per-seed GST reductions.
- `k_efficiency_summary.csv`: k-level means and standard deviations.
- `supplementary_k_efficiency_accuracy.png`: k-level efficiency and accuracy figure.
- `supplementary_alpha_tau_accuracy_efficiency.png`: alpha–tau CV trade-off figure.