"""Run the locked k=45 outer GWSR comparison.

Uses the final standardized prediction guard: non-finite or absolute
standardized predictions at least 5 are replaced by the local adaptive-
bisquare weighted mean in standardized log-response space.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import train_test_split

import original_bisquare as base
import paper_gp
import run_random5fold as protocol

INPUT = Path(os.environ.get("GST_GWSR_STATION_FILE", str(base.STATION_FILE)))
OUT = Path(os.environ.get("GST_GWSR_GWSR_OUTPUT", str(
    Path(__file__).resolve().parents[3] / "run_outputs" / "outer_gwsr")))
K = 45
SEEDS = (42, 123, 456)
OUTER_SEED = 42


def _one(i, xtrain, ytrain, xquery, distances, ys, seed):
    idx, pred, expression, fit_r2, info = paper_gp.predict_one_point(
        i, xtrain, ytrain, xquery, distances, K, seed)
    return dict(idx=int(idx), prediction=float(
        base.inverse_transform(np.asarray([pred]), ys)[0]),
        prediction_scaled=float(pred), expression=expression,
        local_fit_R2=float(fit_r2), **info)


def main() -> None:
    frame = pd.read_csv(INPUT)
    if len(frame) != 1024:
        raise ValueError(f"Expected 1024 rows, found {len(frame)}")
    frame["row_id"] = np.arange(len(frame))
    train_ids, test_ids = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=OUTER_SEED)
    outer_train = frame.iloc[train_ids].copy().reset_index(drop=True)
    outer_test = frame.iloc[test_ids].copy().reset_index(drop=True)
    xtrain, ytrain, xquery, distances, ys, prep = protocol.prepare_training(
        outer_train, outer_test.drop(columns=[base.TARGET]))
    OUT.mkdir(parents=True, exist_ok=True)
    tables, metrics = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            outputs = Parallel(n_jobs=protocol.N_JOBS, backend="loky")(
                delayed(_one)(i, xtrain, ytrain, xquery[i], distances[i], ys, seed)
                for i in range(len(outer_test)))
        outputs.sort(key=lambda x: x["idx"])
        table = outer_test[["row_id", *base.COORDS]].copy()
        table["q_obs"] = outer_test[base.TARGET].to_numpy(dtype=float)
        table["q_pred"] = [x["prediction"] for x in outputs]
        table["residual"] = table.q_obs - table.q_pred
        table["protected"] = [x.get("prediction_protected", False) for x in outputs]
        table["expression"] = [x["expression"] for x in outputs]
        table["seed"] = seed
        table.to_csv(OUT / f"predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
        row = base.metrics(table.q_obs.to_numpy(), table.q_pred.to_numpy(), f"seed_{seed}")
        row.update(seed=seed, seconds=time.perf_counter() - started,
                   protected_count=int(table.protected.sum()))
        metrics.append(row)
        tables.append(table)
    stack = np.stack([t.q_pred.to_numpy() for t in tables])
    obs = tables[0].q_obs.to_numpy()
    ensembles = {}
    for name, pred in (("mean_ensemble", stack.mean(axis=0)),
                       ("median_ensemble", np.median(stack, axis=0))):
        pd.DataFrame({"row_id": tables[0].row_id, "q_obs": obs,
                      "q_pred": pred}).to_csv(OUT / f"predictions_{name}.csv",
                                                index=False, encoding="utf-8-sig")
        ensembles[name] = base.metrics(obs, pred, name)
    pd.DataFrame(metrics).to_csv(OUT / "metrics_by_seed.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "protocol": "locked outer GWSR; k=45; standardized guard",
        "outer_seed": OUTER_SEED, "k": K, "seeds": list(SEEDS),
        "test_response_used_for_model_selection": False,
        "guard": "nonfinite or abs(standardized prediction) >= 5; local weighted mean",
        "inverse_log_clip": [-5, 10], "metrics": metrics,
        "ensembles": ensembles, "preprocessing": prep,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
