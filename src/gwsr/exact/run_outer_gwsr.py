"""Final outer-test GWSR run with a pre-specified training-only guard.

The neighbourhood size k=45 was selected before this run by inner CV.
For each test query and seed, a non-finite or out-of-training-response-range
prediction is replaced by the adaptive-bisquare local weighted training mean.
The rule is applied identically to all seeds and never uses test responses.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve()
SECTION = Path(__file__).resolve().parent
if str(SECTION) not in sys.path:
    sys.path.insert(0, str(SECTION))
import original_bisquare as base
import paper_gp
import run_random5fold as protocol


INPUT = Path(os.environ.get("GST_GWSR_STATION_FILE", str(base.STATION_FILE)))
OUT = Path(os.environ.get(
    "GST_GWSR_GWSR_OUTPUT",
    str(Path(__file__).resolve().parents[3] / "run_outputs" / "outer_gwsr"),
))
K = 45
SEEDS = (42, 123, 456)
OUTER_SEED = 42
GUARD_NAME = "nonfinite_or_outside_clean_training_response_range_to_raw_local_bisquare_mean"


def execute_guarded(i, xtrain, ytrain, xquery, distances, ys, k, seed,
                    log_min, log_max):
    idx, pred_scaled, expression, fit_r2, info = paper_gp.predict_one_point(
        i, xtrain, ytrain, xquery, distances, k, seed
    )
    xloc, yloc_scaled, weights = base.get_local_sample(-1, distances, xtrain, ytrain, k)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_pred = float(ys.inverse_transform(np.asarray([[pred_scaled]]))[0, 0])
        raw_pred = float(np.exp(log_pred))
    protected = (not np.isfinite(raw_pred)) or log_pred < log_min or log_pred > log_max
    # The fallback is defined on the original heat-flow scale.  The GP is
    # fitted to standardized log(q), so convert the local training responses
    # back before taking the bisquare weighted arithmetic mean, then encode
    # that raw fallback in the GP's scaled-log output representation.
    yloc_log = ys.inverse_transform(yloc_scaled.reshape(-1, 1)).ravel()
    yloc_raw = np.exp(yloc_log)
    fallback_raw = float(weights @ yloc_raw)
    fallback_log = float(np.log(fallback_raw))
    fallback_scaled = float(ys.transform(np.asarray([[fallback_log]]))[0, 0])
    if protected:
        pred_scaled_final = fallback_scaled
        expression_final = f"PROTECTED({expression[:120]})"
        log_final = float(ys.inverse_transform(np.asarray([[fallback_scaled]]))[0, 0])
        raw_final = float(np.exp(log_final))
    else:
        pred_scaled_final = float(pred_scaled)
        expression_final = expression
        log_final = log_pred
        raw_final = raw_pred
    return dict(
        idx=int(idx), prediction=float(raw_final), raw_prediction=float(raw_pred),
        prediction_scaled=float(pred_scaled_final), prediction_log=float(log_final),
        raw_prediction_log=float(log_pred), expression=expression_final,
        expression_raw=str(expression), local_fit_R2=float(fit_r2),
        protection_triggered=bool(protected), protection_reason=(
            "outside_clean_training_log_response_range" if protected else ""),
        fallback_prediction=fallback_raw,
        **info,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    if len(frame) != 1024:
        raise ValueError(f"Expected 1024 rows, found {len(frame)}")
    frame["row_id"] = np.arange(len(frame))
    train_ids, test_ids = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=OUTER_SEED
    )
    outer_train = frame.iloc[train_ids].copy().reset_index(drop=True)
    outer_test = frame.iloc[test_ids].copy().reset_index(drop=True)
    prepared = protocol.prepare_training(outer_train, outer_test.drop(columns=[base.TARGET]))
    xtrain, ytrain, xquery, distances, ys, prep_info = prepared
    # Bounds are computed only from the cleaned outer training response.
    log_train = ys.inverse_transform(ytrain.reshape(-1, 1)).ravel()
    log_min, log_max = float(log_train.min()), float(log_train.max())
    raw_min, raw_max = float(np.exp(log_min)), float(np.exp(log_max))
    all_tables = []
    metrics = []
    for seed in SEEDS:
        started = time.perf_counter()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=-1, backend="loky")(
                delayed(execute_guarded)(
                    i, xtrain, ytrain, xquery[i], distances[i], ys, K, seed,
                    log_min, log_max
                ) for i in range(len(outer_test))
            )
        rows.sort(key=lambda r: r["idx"])
        table = outer_test[["row_id", *base.COORDS]].copy()
        table["q_obs"] = outer_test[base.TARGET].to_numpy()
        table["q_pred_raw"] = [r["raw_prediction"] for r in rows]
        table["q_pred"] = [r["prediction"] for r in rows]
        table["residual"] = table.q_obs - table.q_pred
        table["abs_err"] = np.abs(table.residual)
        for key in rows[0]:
            if key != "idx":
                table[key] = [r[key] for r in rows]
        table["k"] = K
        table["seed"] = seed
        table.to_csv(OUT / f"test_k{K}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        row = base.metrics(table.q_obs.to_numpy(), table.q_pred.to_numpy(), f"protected_seed_{seed}")
        row.update(k=K, seed=seed, protection_count=int(table.protection_triggered.sum()),
                   runtime_seconds=time.perf_counter() - started)
        metrics.append(row)
        all_tables.append(table)
        print(json.dumps(row), flush=True)

    stack = np.stack([t.q_pred.to_numpy() for t in all_tables])
    y = all_tables[0].q_obs.to_numpy()
    ensembles = {}
    for name, pred in (("mean_ensemble", stack.mean(axis=0)),
                       ("median_ensemble", np.median(stack, axis=0))):
        row = base.metrics(y, pred, name)
        row.update(k=K, seed=-1)
        ensembles[name] = row
        pd.DataFrame({"row_id": outer_test.row_id, "q_obs": y,
                      "q_pred": pred}).to_csv(
            OUT / f"test_k{K}_{name}.csv", index=False, encoding="utf-8-sig"
        )
        print(json.dumps(row), flush=True)

    manifest = dict(
        protocol="strict outer test with pre-specified prediction protection",
        n_original=len(frame), n_outer_train_raw=len(outer_train),
        n_outer_train_clean=len(xtrain), n_outer_test=len(outer_test),
        split_seed=OUTER_SEED, k=K, seeds=list(SEEDS), kernel="adaptive_bisquare",
        test_response_used_for_selection=False,
        guard_rule=GUARD_NAME, clean_training_raw_response_min=raw_min,
        clean_training_raw_response_max=raw_max,
        clean_training_log_response_min=log_min,
        clean_training_log_response_max=log_max,
        metrics=metrics, ensembles=ensembles, preprocessing=prep_info,
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(OUT / "metrics_by_seed.csv", index=False)
    (OUT / "ensemble_metrics.json").write_text(json.dumps(ensembles, indent=2), encoding="utf-8")
    print(f"All protected results: {OUT}", flush=True)


if __name__ == "__main__":
    main()
