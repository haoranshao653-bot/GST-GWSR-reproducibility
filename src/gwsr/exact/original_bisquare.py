"""
Parallel GW-SR with outliers removed from the training set only.

This version is directly comparable with the original-data GW-SR results:
1. Split the original 1024 samples with the same 8:2 random_state=42 split.
2. Keep the original test set unchanged.
3. Remove 3*IQR heat-flow outliers only from the training set.
4. Run the original GW-SR settings with CPU parallelism, multi-k, and multi-seed.
"""

import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


REPO_ROOT = Path(__file__).resolve().parents[3]
STATION_FILE = Path(
    os.environ.get("GST_GWSR_STATION_FILE", str(REPO_ROOT / "data" / "processed" / "heat_flow_stations.csv"))
)
BASE_OUTPUT_DIR = Path(
    os.environ.get("GST_GWSR_OUTPUT_DIR", str(REPO_ROOT / "run_outputs"))
)
RUN_NAME = "parallel_train_cleaned_3iqr_bisquare_kernel_only"
OUTPUT_DIR = BASE_OUTPUT_DIR / RUN_NAME

FEATURES = ["topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano"]
TARGET = "heat_flow"
COORDS = ["lon", "lat"]

BW_LIST = [25, 30, 35, 40]
SEEDS = [42, 123, 456]
N_JOBS = -1


def iqr_bounds(values: pd.Series, multiplier: float = 3.0) -> Tuple[float, float, float, float, float]:
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return q1, q3, iqr, lower, upper


def get_local_sample(
    center_idx: int,
    d_row: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    k: int,
    min_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_use = max(k, min_samples)
    k_use = min(k_use, len(d_row))
    sorted_idx = np.argsort(d_row)
    if center_idx >= 0:
        sorted_idx = sorted_idx[sorted_idx != center_idx]

    local_idx = sorted_idx[:k_use]
    x_local = x_ref[local_idx]
    y_local = y_ref[local_idx]
    d_local = d_row[local_idx]
    h = d_row[sorted_idx[k_use - 1]] + 1e-8
    weights = np.where(d_local < h, (1.0 - (d_local / h) ** 2) ** 2, 0.0)
    weights = weights / weights.sum()
    return x_local, y_local, weights


def build_sr_model(seed: int) -> SymbolicRegressor:
    return SymbolicRegressor(
        population_size=1000,
        generations=20,
        tournament_size=20,
        init_depth=(2, 5),
        init_method="half and half",
        function_set=["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"],
        metric="mse",
        parsimony_coefficient=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.8,
        feature_names=FEATURES,
        warm_start=False,
        n_jobs=1,
        verbose=0,
        random_state=seed,
    )


def predict_one_point(
    i: int,
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    x_pred_row: np.ndarray,
    d_row: np.ndarray,
    k: int,
    seed: int,
) -> Tuple[int, float, str, float]:
    x_loc, y_loc, weights = get_local_sample(-1, d_row, x_train, y_train_scaled, k)
    sr = build_sr_model(seed + i)

    try:
        sr.fit(x_loc, y_loc, sample_weight=weights)
        pred_scaled = float(sr.predict(x_pred_row.reshape(1, -1))[0])
        expr = str(sr._program)
        fit_r2 = float(r2_score(y_loc, sr.predict(x_loc)))

        if not np.isfinite(pred_scaled) or abs(pred_scaled) >= 5:
            pred_scaled = float(np.average(y_loc, weights=weights))
            expr = f"CLIPPED({expr[:80]})"
    except Exception as error:
        pred_scaled = float(np.average(y_loc, weights=weights))
        expr = f"FAILED({str(error)[:80]})"
        fit_r2 = 0.0

    return i, pred_scaled, expr, fit_r2


def inverse_transform(y_scaled: np.ndarray, scaler_y: StandardScaler) -> np.ndarray:
    y_log = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    return np.exp(np.clip(y_log, -5, 10))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return len(y_true) * np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true)) / np.sum(y_true)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "model": label,
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "N_RMSE": nrmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def print_metrics(row: Dict[str, float]) -> None:
    print(
        f"  [{row['model']}] R2={row['R2']:.4f}  RMSE={row['RMSE']:.3f}  "
        f"N-RMSE={row['N_RMSE']:.4f}  MAE={row['MAE']:.3f}  MAPE={row['MAPE']:.2f}%"
    )


def prediction_column_name(model_name: str) -> str:
    return "q_pred_" + model_name.replace("=", "")


def prediction_path(k: int, seed: int) -> Path:
    return OUTPUT_DIR / f"{RUN_NAME}_pred_k{k}_seed{seed}.csv"


def load_prediction(path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    cached = pd.read_csv(path)
    y_pred = cached["q_pred"].values
    expressions = cached["expression"].fillna("").astype(str).tolist()
    fit_scores = cached.get("local_fit_R2", pd.Series([np.nan] * len(cached))).values
    return y_pred, expressions, fit_scores


def save_prediction(
    path: Path,
    frame: pd.DataFrame,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    expressions: List[str],
    fit_scores: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "lon": frame["lon"].values,
            "lat": frame["lat"].values,
            "q_obs": y_obs,
            "q_pred": y_pred,
            "residual": y_obs - y_pred,
            "abs_err": np.abs(y_obs - y_pred),
            "local_fit_R2": fit_scores,
            "expression": expressions,
        }
    ).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def run_parallel_prediction(
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    x_test: np.ndarray,
    d_test: np.ndarray,
    scaler_y: StandardScaler,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    k: int,
    seed: int,
) -> Tuple[np.ndarray, List[str], np.ndarray, float]:
    path = prediction_path(k, seed)
    if path.exists():
        print(f"\n  loading cached: {path.name}")
        y_pred, exprs, fit_scores = load_prediction(path)
        return y_pred, exprs, fit_scores, 0.0

    print(f"\n  running k={k}, seed={seed}, n_jobs={N_JOBS}")
    start = time.time()
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(predict_one_point)(
            i,
            x_train,
            y_train_scaled,
            x_test[i],
            d_test[i],
            k,
            seed,
        )
        for i in range(len(x_test))
    )
    elapsed = (time.time() - start) / 60

    results = sorted(results, key=lambda item: item[0])
    pred_scaled = np.array([item[1] for item in results])
    expressions = [item[2] for item in results]
    fit_scores = np.array([item[3] for item in results])
    y_pred = inverse_transform(pred_scaled, scaler_y)

    save_prediction(path, test_df, y_test, y_pred, expressions, fit_scores)
    print(f"  finished in {elapsed:.2f} min")
    print(f"  saved: {path.name}")
    return y_pred, expressions, fit_scores, elapsed


def save_bandwidth_plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for model_type, group in summary.groupby("type"):
        ax.plot(group["k"], group["R2"], "o-", label=model_type, lw=2)
    ax.set_xlabel("Bandwidth k")
    ax.set_ylabel("Test R2")
    ax.set_title("Training-cleaned GW-SR")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    for model_type, group in summary.groupby("type"):
        ax.plot(group["k"], group["RMSE"], "s-", label=model_type, lw=2)
    ax.set_xlabel("Bandwidth k")
    ax.set_ylabel("Test RMSE")
    ax.set_title("RMSE by bandwidth")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{RUN_NAME}_06_bandwidth_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("South China Sea heat-flow prediction - train-only 3*IQR cleaned GW-SR")
    print("=" * 72)
    print(f"Input file: {STATION_FILE}")
    print(f"Output folder: {OUTPUT_DIR}")

    df = pd.read_csv(STATION_FILE)
    x_raw = df[FEATURES].values
    y_all = df[TARGET].values
    coords = df[COORDS].values

    idx = np.arange(len(df))
    idx_train_raw, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    train_raw_df = df.iloc[idx_train_raw].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)

    q1, q3, iqr, lower, upper = iqr_bounds(train_raw_df[TARGET], multiplier=3.0)
    keep_train_mask = (train_raw_df[TARGET] >= lower) & (train_raw_df[TARGET] <= upper)
    removed_train_df = train_raw_df[~keep_train_mask].copy()
    train_df = train_raw_df[keep_train_mask].reset_index(drop=True)

    removed_train_path = OUTPUT_DIR / f"{RUN_NAME}_00_removed_train_outliers.csv"
    removed_train_df.to_csv(removed_train_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    x_train_raw = train_df[FEATURES].values
    x_test_raw = test_df[FEATURES].values
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values
    coords_train = train_df[COORDS].values
    coords_test = test_df[COORDS].values

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train_raw)
    x_test = scaler_x.transform(x_test_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(np.log(y_train).reshape(-1, 1)).ravel()

    d_test = cdist(coords_test, coords_train, metric="euclidean")

    print(f"Original samples: all={len(df)}, raw_train={len(train_raw_df)}, test={len(test_df)}")
    print(f"Train-only 3*IQR bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"Removed from training: {len(removed_train_df)}")
    print(f"Final train={len(train_df)}, unchanged test={len(test_df)}")
    if len(removed_train_df):
        print("Removed training heat_flow values:")
        print(removed_train_df[[*COORDS, TARGET]].sort_values(TARGET, ascending=False).to_string(index=False))
    print(f"Test heat flow range remains: [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"Bandwidths: {BW_LIST}")
    print(f"Seeds: {SEEDS}")
    print(f"Parallel workers: {N_JOBS}")

    rows: List[Dict[str, float]] = []
    predictions: Dict[Tuple[int, int], np.ndarray] = {}
    expressions: Dict[Tuple[int, int], List[str]] = {}
    fit_r2_by_run: Dict[Tuple[int, int], np.ndarray] = {}

    compare_df = pd.DataFrame(
        {
            "lon": test_df["lon"].values,
            "lat": test_df["lat"].values,
            "q_obs": y_test,
        }
    )

    for k in BW_LIST:
        print(f"\n--- Bandwidth k={k} ---")
        for seed in SEEDS:
            y_pred, exprs, fit_scores, elapsed = run_parallel_prediction(
                x_train,
                y_train_scaled,
                x_test,
                d_test,
                scaler_y,
                test_df,
                y_test,
                k,
                seed,
            )
            row = metrics(y_test, y_pred, f"k={k}_seed={seed}")
            row["k"] = k
            row["seed"] = seed
            row["type"] = "single_seed"
            row["time_min"] = elapsed
            row["mean_local_fit_R2"] = float(np.nanmean(fit_scores))
            rows.append(row)
            print_metrics(row)
            print(f"    mean local fit R2={row['mean_local_fit_R2']:.4f}")

            predictions[(k, seed)] = y_pred
            expressions[(k, seed)] = exprs
            fit_r2_by_run[(k, seed)] = fit_scores
            compare_df[prediction_column_name(f"k={k}_seed={seed}")] = y_pred

        pred_stack = np.vstack([predictions[(k, seed)] for seed in SEEDS])
        pred_mean = pred_stack.mean(axis=0)
        pred_median = np.median(pred_stack, axis=0)

        for ensemble_type, pred in [("mean_ensemble", pred_mean), ("median_ensemble", pred_median)]:
            row = metrics(y_test, pred, f"k={k}_{ensemble_type}")
            row["k"] = k
            row["seed"] = -1
            row["type"] = ensemble_type
            row["time_min"] = 0.0
            row["mean_local_fit_R2"] = np.nan
            rows.append(row)
            print_metrics(row)
            compare_df[prediction_column_name(f"k={k}_{ensemble_type}")] = pred

    metrics_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    metrics_path = OUTPUT_DIR / f"{RUN_NAME}_01_metrics_all.csv"
    compare_path = OUTPUT_DIR / f"{RUN_NAME}_02_test_prediction_compare.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    best = metrics_df.iloc[0]
    best_model = str(best["model"])
    best_k = int(best["k"])
    best_col = prediction_column_name(best_model)
    best_pred = compare_df[best_col].values

    print("\n" + "=" * 72)
    print(f"Best model: {best_model}")
    print(f"Best R2={best['R2']:.4f}, RMSE={best['RMSE']:.3f}, MAE={best['MAE']:.3f}")
    print("=" * 72)

    if "mean_ensemble" in best_model or "median_ensemble" in best_model:
        best_exprs = expressions[(best_k, SEEDS[0])]
        best_fit_scores = fit_r2_by_run[(best_k, SEEDS[0])]
    else:
        best_seed = int(best["seed"])
        best_exprs = expressions[(best_k, best_seed)]
        best_fit_scores = fit_r2_by_run[(best_k, best_seed)]

    pd.DataFrame(
        {
            "lon": test_df["lon"].values,
            "lat": test_df["lat"].values,
            "q_obs": y_test,
            "q_pred": best_pred,
            "residual": y_test - best_pred,
            "abs_err": np.abs(y_test - best_pred),
            "local_fit_R2_reference": best_fit_scores,
            "expression_reference": best_exprs,
        }
    ).to_csv(
        OUTPUT_DIR / f"{RUN_NAME}_03_best_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    metrics_df.to_csv(OUTPUT_DIR / f"{RUN_NAME}_04_summary_by_k.csv", index=False, encoding="utf-8-sig")
    save_bandwidth_plot(metrics_df)

    with (OUTPUT_DIR / f"{RUN_NAME}_07_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "station_file": str(STATION_FILE),
                "features": FEATURES,
                "bandwidths": BW_LIST,
                "seeds": SEEDS,
                "n_jobs": N_JOBS,
                "raw_train_count": int(len(train_raw_df)),
                "removed_train_count": int(len(removed_train_df)),
                "final_train_count": int(len(train_df)),
                "unchanged_test_count": int(len(test_df)),
                "train_outlier_bounds_3iqr": {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower": lower,
                    "upper": upper,
                },
                "best_model": best_model,
                "best_R2": float(best["R2"]),
                "best_RMSE": float(best["RMSE"]),
                "note": "Original 8:2 split retained; only training-set 3*IQR outliers removed; test set unchanged.",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {removed_train_path}")
    print(f"  {metrics_path}")
    print(f"  {compare_path}")
    print(f"  {RUN_NAME}_03_best_test_predictions.csv")
    print(f"  {RUN_NAME}_04_summary_by_k.csv")
    print(f"  {RUN_NAME}_06_bandwidth_summary.png")
    print(f"  {RUN_NAME}_07_config.json")
    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
