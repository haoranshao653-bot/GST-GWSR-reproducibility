"""
Training-only spatial selection of the GWSR/GST-GWSR neighbourhood size k.

This script is deliberately separate from gst_gwsr_core.py.  It selects k on
the OUTER training partition only, then writes the selected k for the final
GST-GWSR run.  The held-out outer test targets are never read by the selector.

To keep selection practical, the criterion is a geographically weighted local
mean proxy rather than a complete symbolic-regression/GST run for every k.
It isolates the spatial bandwidth decision and matches the intended original
robustness procedure:

  candidate k -> inner spatial blocks -> Gaussian local weighted mean
              -> pooled transformed-target RMSE -> select the smallest RMSE.

Default candidates: 15,20,25,30,40,50; default inner spatial folds: 5.

Example:
python select_gwsr_k_training_only.py ^
  --input heat_flow_stations.csv --target heat_flow --lon lon --lat lat ^
  --features topo,bouguer,sediment,moho,lab,cpd,lithos_km,volcano
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


EPSILON = 1e-8
EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select GWSR k using outer-training-only spatial validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Labelled input CSV.")
    parser.add_argument("--target", required=True, help="Target column.")
    parser.add_argument("--lon", default="lon", help="Longitude/X column.")
    parser.add_argument("--lat", default="lat", help="Latitude/Y column.")
    parser.add_argument(
        "--features",
        default="",
        help="Optional comma-separated predictors; validated but not used by the local-mean proxy.",
    )
    parser.add_argument("--output", default="", help="Output folder.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--iqr-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--target-transform",
        choices=("auto", "log", "none"),
        default="auto",
    )
    parser.add_argument(
        "--distance-metric",
        choices=("euclidean_degrees", "haversine_km"),
        default="euclidean_degrees",
    )
    parser.add_argument(
        "--k-candidates",
        default="15,20,25,30,40,50",
        help="Comma-separated candidate neighbour counts.",
    )
    parser.add_argument("--inner-spatial-folds", type=int, default=5)
    return parser.parse_args()


def parse_k_candidates(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--k-candidates must be comma-separated integers.") from exc
    if not values or min(values) < 2 or len(set(values)) != len(values):
        raise ValueError("Candidate k values must be unique integers no smaller than 2.")
    return tuple(sorted(values))


def resolve_features(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[str, ...]:
    if args.features.strip():
        features = tuple(item.strip() for item in args.features.split(",") if item.strip())
    else:
        excluded = {args.target, args.lon, args.lat}
        features = tuple(
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in excluded
        )
    required = [args.target, args.lon, args.lat, *features]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")
    return features


def load_complete_rows(
    path: Path,
    target: str,
    lon: str,
    lat: str,
    features: tuple[str, ...],
) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path)
    needed = [target, lon, lat, *features]
    data = frame.loc[:, needed].copy()
    data.insert(0, "source_row", frame.index.to_numpy())
    for column in needed:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = np.isfinite(data[needed].to_numpy(dtype=float)).all(axis=1)
    return data.loc[keep].reset_index(drop=True), int((~keep).sum())


def haversine_km(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    lon_a = np.radians(first[:, 0])[:, None]
    lat_a = np.radians(first[:, 1])[:, None]
    lon_b = np.radians(second[:, 0])[None, :]
    lat_b = np.radians(second[:, 1])[None, :]
    a = (
        np.sin((lat_a - lat_b) / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_a - lon_b) / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pairwise_distance(query: np.ndarray, train: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean_degrees":
        return cdist(query, train, metric="euclidean")
    return haversine_km(query, train)


def spatial_blocks(frame: pd.DataFrame, lon: str, lat: str, n_blocks: int, seed: int) -> np.ndarray:
    """K-means spatial blocks, ordered deterministically by centroid location."""
    mean_lat = np.deg2rad(float(frame[lat].mean()))
    coordinates_km = np.column_stack(
        [
            frame[lon].to_numpy(dtype=float) * 111.32 * np.cos(mean_lat),
            frame[lat].to_numpy(dtype=float) * 110.57,
        ]
    )
    raw = KMeans(n_clusters=n_blocks, random_state=seed, n_init=50).fit_predict(coordinates_km)
    centroids = []
    for label in range(n_blocks):
        subset = frame.iloc[np.flatnonzero(raw == label)]
        centroids.append((label, float(subset[lon].mean()), float(subset[lat].mean())))
    mapping = {
        old: new
        for new, (old, _, _) in enumerate(sorted(centroids, key=lambda item: (item[1], item[2])))
    }
    return np.asarray([mapping[int(label)] for label in raw], dtype=int)


def resolve_target_transform(values: np.ndarray, requested: str) -> str:
    if requested == "auto":
        return "log" if np.all(values > 0) else "none"
    if requested == "log" and not np.all(values > 0):
        raise ValueError("--target-transform log requires all retained training targets to be positive.")
    return requested


def inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    return np.exp(np.clip(values, -30.0, 30.0)) if transform == "log" else values


def run_selection(args: argparse.Namespace) -> tuple[int, pd.DataFrame, dict[str, Any]]:
    input_path = Path(args.input).expanduser().resolve()
    raw = pd.read_csv(input_path)
    features = resolve_features(raw, args)
    data, dropped = load_complete_rows(input_path, args.target, args.lon, args.lat, features)
    if len(data) < 20:
        raise ValueError("At least 20 complete rows are required.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")

    positions = np.arange(len(data))
    outer_train_positions, outer_test_positions = train_test_split(
        positions, test_size=args.test_size, random_state=args.split_seed
    )
    outer_train_raw = data.iloc[outer_train_positions].reset_index(drop=True)
    # The next line is intentionally never used in calculation; retaining the
    # variable makes the outer-test exclusion explicit to readers of the code.
    outer_test_unused = data.iloc[outer_test_positions].reset_index(drop=True)
    del outer_test_unused

    raw_target = outer_train_raw[args.target].to_numpy(dtype=float)
    if args.iqr_multiplier > 0:
        q1, q3 = np.quantile(raw_target, [0.25, 0.75])
        iqr = float(q3 - q1)
        lower = float(q1 - args.iqr_multiplier * iqr)
        upper = float(q3 + args.iqr_multiplier * iqr)
        keep = outer_train_raw[args.target].between(lower, upper).to_numpy()
    else:
        lower, upper = float("-inf"), float("inf")
        keep = np.ones(len(outer_train_raw), dtype=bool)
    outer_train = outer_train_raw.loc[keep].reset_index(drop=True)
    if len(outer_train) < 12:
        raise ValueError("Too few outer-training rows remain after IQR cleaning.")

    transform = resolve_target_transform(
        outer_train[args.target].to_numpy(dtype=float), args.target_transform
    )
    candidate_k = parse_k_candidates(args.k_candidates)
    n_blocks = min(args.inner_spatial_folds, len(outer_train))
    if n_blocks < 2:
        raise ValueError("At least two inner spatial folds are required.")
    labels = spatial_blocks(outer_train, args.lon, args.lat, n_blocks, args.split_seed)

    pooled: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {k: [] for k in candidate_k}
    rows: list[dict[str, Any]] = []
    valid_candidates = set(candidate_k)
    for fold in range(n_blocks):
        validation = outer_train.iloc[np.flatnonzero(labels == fold)]
        inner_train = outer_train.iloc[np.flatnonzero(labels != fold)]
        usable = {k for k in candidate_k if k <= len(inner_train)}
        valid_candidates.intersection_update(usable)
        if not usable or validation.empty:
            continue
        distance = pairwise_distance(
            validation[[args.lon, args.lat]].to_numpy(dtype=float),
            inner_train[[args.lon, args.lat]].to_numpy(dtype=float),
            args.distance_metric,
        )
        order = np.argsort(distance, axis=1)
        train_raw_y = inner_train[args.target].to_numpy(dtype=float)
        observed_raw = validation[args.target].to_numpy(dtype=float)
        train_y = np.log(train_raw_y) if transform == "log" else train_raw_y
        observed_y = np.log(observed_raw) if transform == "log" else observed_raw
        for k in sorted(usable):
            indices = order[:, :k]
            local_distance = np.take_along_axis(distance, indices, axis=1)
            bandwidth = np.maximum(local_distance[:, -1:], EPSILON)
            weights = np.exp(-0.5 * (local_distance / bandwidth) ** 2)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPSILON)
            predicted_y = np.sum(weights * train_y[indices], axis=1)
            predicted_raw = inverse_target(predicted_y, transform)
            pooled[k].append((observed_y, predicted_y))
            rows.append(
                {
                    "record_type": "inner_spatial_fold",
                    "inner_fold": fold,
                    "k_neighbors": k,
                    "sample_count": int(len(validation)),
                    "selection_scale": transform,
                    "transformed_rmse": float(np.sqrt(mean_squared_error(observed_y, predicted_y))),
                    "raw_rmse": float(np.sqrt(mean_squared_error(observed_raw, predicted_raw))),
                    "raw_mae": float(mean_absolute_error(observed_raw, predicted_raw)),
                    "selected": False,
                }
            )
    if not valid_candidates:
        raise ValueError("No candidate k is small enough for every inner training fold.")

    aggregate_rows: list[dict[str, Any]] = []
    for k in sorted(valid_candidates):
        observed_y = np.concatenate([values[0] for values in pooled[k]])
        predicted_y = np.concatenate([values[1] for values in pooled[k]])
        observed_raw = inverse_target(observed_y, transform)
        predicted_raw = inverse_target(predicted_y, transform)
        aggregate_rows.append(
            {
                "record_type": "pooled_inner_validation",
                "inner_fold": -1,
                "k_neighbors": k,
                "sample_count": int(len(observed_y)),
                "selection_scale": transform,
                "transformed_rmse": float(np.sqrt(mean_squared_error(observed_y, predicted_y))),
                "raw_rmse": float(np.sqrt(mean_squared_error(observed_raw, predicted_raw))),
                "raw_mae": float(mean_absolute_error(observed_raw, predicted_raw)),
                "selected": False,
            }
        )
    best = min(aggregate_rows, key=lambda item: (item["transformed_rmse"], item["k_neighbors"]))
    selected_k = int(best["k_neighbors"])
    for row in aggregate_rows:
        row["selected"] = row["k_neighbors"] == selected_k

    report = pd.DataFrame(rows + aggregate_rows)
    summary = {
        "input_csv": str(input_path),
        "target": args.target,
        "coordinates": [args.lon, args.lat],
        "features": list(features),
        "outer_test_size": args.test_size,
        "outer_split_seed": args.split_seed,
        "outer_training_rows_before_iqr": int(len(outer_train_raw)),
        "outer_training_rows_after_iqr": int(len(outer_train)),
        "outer_training_rows_removed_by_iqr": int((~keep).sum()),
        "outer_test_rows_excluded_from_selection": int(len(outer_test_positions)),
        "missing_or_non_numeric_rows_removed_before_split": dropped,
        "training_iqr_lower": lower,
        "training_iqr_upper": upper,
        "target_transform_used": transform,
        "distance_metric": args.distance_metric,
        "selection_proxy": "Gaussian geographically weighted local mean",
        "inner_spatial_folds": n_blocks,
        "candidate_k": list(candidate_k),
        "selected_k": selected_k,
        "selection_criterion": "minimum pooled transformed-target RMSE; ties select smaller k",
        "test_target_values_used": False,
    }
    return selected_k, report, summary


def main() -> None:
    args = parse_args()
    selected_k, report, summary = run_selection(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else input_path.with_name(
        f"{input_path.stem}_gwsr_k_selection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "gwsr_k_selection_results.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "gwsr_k_selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    pooled = report.loc[report["record_type"] == "pooled_inner_validation"].sort_values("k_neighbors")
    print("=" * 76)
    print("Training-only spatial selection of GWSR/GST-GWSR k")
    print("=" * 76)
    print(f"Outer-test target values used: NO (excluded rows: {summary['outer_test_rows_excluded_from_selection']})")
    print(f"Outer training rows after IQR cleaning: {summary['outer_training_rows_after_iqr']}")
    print(f"Selection proxy: {summary['selection_proxy']}")
    print("\nPooled inner-spatial validation:")
    print(pooled[["k_neighbors", "transformed_rmse", "raw_rmse", "raw_mae", "selected"]].to_string(index=False))
    print(f"\nSelected k: {selected_k}")
    print(f"Use this in the final GST-GWSR run: --k-neighbors {selected_k}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nK selection stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
