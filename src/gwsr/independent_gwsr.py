"""Independent geographically weighted symbolic regression.

This script implements the GWSR baseline only.  It intentionally contains no
GST, warm starts, task association, population transfer, or alpha parameter.

Protocol used by the current marine heat-flow experiment
---------------------------------------------------------
* random_state=42 fixed 80:20 split (819 outer-training, 205 evaluation rows)
* 3-IQR filtering is applied to the outer-training response only
* predictors are standardized with the retained training rows
* log(response) is standardized with the retained training rows
* each local task uses the k nearest retained training rows and adaptive
  bisquare weights inside that neighbourhood
* GP settings match the reported independent GWSR baseline

The evaluation response is read only after all local searches finish.  Passing
--k-grid runs a descriptive sensitivity analysis; it never chooses a k.
The primary experiment uses --k 35, matching the fixed-split protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = ["topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano"]
COORDS = ["lon", "lat"]
TARGET = "heat_flow"
DEFAULT_DATA = Path(os.environ.get(
    "GST_GWSR_STATION_FILE",
    str(Path(__file__).resolve().parents[2] / "data" / "processed" / "heat_flow_stations.csv"),
))


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 1000
    generations: int = 20
    tournament_size: int = 20
    init_depth_min: int = 2
    init_depth_max: int = 5
    parsimony_coefficient: float = 0.01
    p_crossover: float = 0.70
    p_subtree_mutation: float = 0.10
    p_hoist_mutation: float = 0.05
    p_point_mutation: float = 0.10
    max_samples: float = 0.80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent adaptive-bisquare GWSR baseline")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=Path("gwsr_independent_results"))
    p.add_argument("--k", type=int, default=35, help="Primary fixed neighbourhood size")
    p.add_argument("--k-grid", default="", help="Optional comma-separated sensitivity grid; no automatic selection")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--prediction-guard", type=float, default=5.0,
                   help="Fallback to the local weighted mean when |scaled prediction| exceeds this value")
    return p.parse_args()


def iqr_filter(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    q1, q3 = train[TARGET].quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    bounds = {"q1": float(q1), "q3": float(q3), "iqr": iqr,
              "lower": float(q1 - 3.0 * iqr), "upper": float(q3 + 3.0 * iqr)}
    keep = train[TARGET].between(bounds["lower"], bounds["upper"])
    return train.loc[keep].copy(), train.loc[~keep].copy(), bounds


def local_task(x_train: np.ndarray, y_train: np.ndarray, train_coords: np.ndarray,
               query_coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the k nearest training rows and normalized adaptive-bisquare weights."""
    k = min(max(int(k), 1), len(x_train))
    distances = cdist(query_coords, train_coords, metric="euclidean")[0]
    nearest = np.argsort(distances, kind="mergesort")[:k]
    local_dist = distances[nearest]
    bandwidth = float(local_dist[-1])
    if bandwidth <= np.finfo(float).eps:
        weights = np.ones(k, dtype=float)
    else:
        ratio = local_dist / (bandwidth + 1e-12)
        weights = np.where(ratio < 1.0, (1.0 - ratio**2) ** 2, 0.0)
        if weights.sum() <= 0.0:
            weights = np.ones(k, dtype=float)
    weights /= weights.sum()
    return x_train[nearest], y_train[nearest], weights, bandwidth


def build_gp(seed: int, config: GPConfig, feature_names: list[str]) -> SymbolicRegressor:
    return SymbolicRegressor(
        population_size=config.population_size,
        generations=config.generations,
        tournament_size=config.tournament_size,
        init_depth=(config.init_depth_min, config.init_depth_max),
        init_method="half and half",
        function_set=["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"],
        metric="mse",
        parsimony_coefficient=config.parsimony_coefficient,
        p_crossover=config.p_crossover,
        p_subtree_mutation=config.p_subtree_mutation,
        p_hoist_mutation=config.p_hoist_mutation,
        p_point_mutation=config.p_point_mutation,
        p_point_replace=0.05,
        max_samples=config.max_samples,
        feature_names=feature_names,
        n_jobs=1,
        random_state=seed,
        verbose=0,
    )


def fit_local(task_id: int, x_train: np.ndarray, y_train: np.ndarray,
               train_coords: np.ndarray, query_coords: np.ndarray, k: int,
               query_x: np.ndarray, seed: int, guard: float, config: GPConfig) -> dict:
    x_local, y_local, weights, bandwidth = local_task(
        x_train, y_train, train_coords, query_coords[task_id:task_id + 1], k
    )
    model = build_gp(seed + task_id, config, FEATURES)
    local_mean = float(np.average(y_local, weights=weights))
    guard_triggered = False
    failure = ""
    try:
        model.fit(x_local, y_local, sample_weight=weights)
        pred = float(model.predict(query_x[task_id:task_id + 1])[0])
        expression = str(model._program)
        if not np.isfinite(pred) or abs(pred) >= guard:
            pred, guard_triggered = local_mean, True
            expression = f"GUARDED({expression})"
    except Exception as exc:  # keep one failed local task from discarding the run
        pred, guard_triggered = local_mean, True
        expression, failure = f"FAILED({type(exc).__name__})", str(exc)
    return {"task_id": task_id, "prediction_scaled": pred, "expression": expression,
            "bandwidth": bandwidth, "guard_triggered": guard_triggered, "failure": failure,
            "local_sample_count": len(y_local)}


def inverse_target(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    log_values = scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()
    return np.exp(np.clip(log_values, -5.0, 10.0))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, k: int, seed: int, seconds: float) -> dict:
    return {"k": k, "seed": seed, "R2": r2_score(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE_percent": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "seconds": seconds}


def run_one(k: int, seed: int, x_train: np.ndarray, y_train: np.ndarray,
            train_coords: np.ndarray, x_eval: np.ndarray, eval_coords: np.ndarray,
            y_eval: np.ndarray, y_scaler: StandardScaler, output: Path,
            guard: float, config: GPConfig, n_jobs: int) -> tuple[dict, pd.DataFrame]:
    start = time.perf_counter()
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_local)(i, x_train, y_train, train_coords, eval_coords, k, x_eval, seed, guard, config)
        for i in range(len(x_eval))
    )
    seconds = time.perf_counter() - start
    table = pd.DataFrame(sorted(rows, key=lambda row: row["task_id"]))
    table["q_pred"] = inverse_target(table["prediction_scaled"].to_numpy(), y_scaler)
    table["q_obs"] = y_eval
    table["residual"] = table["q_pred"] - y_eval
    table["abs_error"] = np.abs(table["residual"])
    table["k"], table["seed"] = k, seed
    table.to_csv(output / f"gwsr_k{k}_seed{seed}_predictions.csv", index=False, encoding="utf-8-sig")
    return metric_row(y_eval, table["q_pred"].to_numpy(), k, seed, seconds), table


def main() -> None:
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    k_values = [args.k] if not args.k_grid.strip() else [int(x) for x in args.k_grid.split(",")]
    if args.k not in k_values:
        k_values.append(args.k)
    if any(k < 1 for k in k_values):
        raise ValueError("All k values must be positive")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    required = FEATURES + COORDS + [TARGET]
    missing = [c for c in required if c not in df]
    if missing: raise ValueError(f"Missing columns: {missing}")
    indices = np.arange(len(df))
    train_idx, eval_idx = train_test_split(indices, test_size=args.test_size, random_state=args.split_seed)
    train_raw, evaluation = df.iloc[train_idx].copy(), df.iloc[eval_idx].copy()
    train, removed, bounds = iqr_filter(train_raw)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(train[FEATURES]); x_eval = x_scaler.transform(evaluation[FEATURES])
    y_train = y_scaler.fit_transform(np.log(train[TARGET]).to_numpy().reshape(-1, 1)).ravel()
    train_coords, eval_coords = train[COORDS].to_numpy(), evaluation[COORDS].to_numpy()
    y_eval = evaluation[TARGET].to_numpy(dtype=float)
    config = GPConfig()
    train.assign(split="training", retained_after_iqr=True).to_csv(output / "training_rows.csv", index=False)
    evaluation.assign(split="evaluation").to_csv(output / "evaluation_rows.csv", index=False)
    removed.assign(split="training_removed_by_3IQR").to_csv(output / "removed_training_rows.csv", index=False)
    metrics, primary_tables = [], {}
    for k in k_values:
        seed_tables = []
        for seed in seeds:
            row, table = run_one(k, seed, x_train, y_train, train_coords, x_eval, eval_coords,
                                 y_eval, y_scaler, output, args.prediction_guard, config, args.n_jobs)
            metrics.append(row); seed_tables.append(table)
        stack = np.vstack([t["q_pred"].to_numpy() for t in seed_tables])
        ensemble = stack.mean(axis=0)
        metrics.append({"k": k, "seed": "mean_ensemble", "R2": r2_score(y_eval, ensemble),
                        "RMSE": mean_squared_error(y_eval, ensemble, squared=False),
                        "MAE": mean_absolute_error(y_eval, ensemble),
                        "MAPE_percent": float(np.mean(np.abs((y_eval - ensemble) / y_eval)) * 100),
                        "seconds": float(sum(row["seconds"] for row in metrics[-len(seeds):]))})
        if k == args.k: primary_tables["ensemble_prediction"] = ensemble
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"q_obs": y_eval, "q_pred_ensemble": primary_tables["ensemble_prediction"]}).to_csv(
        output / f"gwsr_k{args.k}_ensemble_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {"method": "independent GWSR", "data": str(args.data), "features": FEATURES,
                "split_seed": args.split_seed, "test_size": args.test_size,
                "raw_training_count": len(train_raw), "retained_training_count": len(train),
                "evaluation_count": len(evaluation), "removed_training_count": len(removed),
                "iqr_bounds": bounds, "k_primary": args.k, "k_grid": k_values, "seeds": seeds,
                "kernel": "adaptive bisquare within the k nearest retained training rows",
                "evaluation_targets_used_for_fitting": False,
                "k_selection": "not performed by this script; k is supplied explicitly",
                "gp": asdict(config), "prediction_guard": args.prediction_guard}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
