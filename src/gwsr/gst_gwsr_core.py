"""
Portable GST-GWSR core implementation
======================================

This is a single-file implementation of Geographic-Semantic Transfer enhanced
Geographically Weighted Symbolic Regression (GST-GWSR).  It keeps the core
workflow used by the original project while removing experiment-specific paths,
plots, ablation code, LLM calls, bootstrap code, and intermediate caches.

Core workflow
-------------
1. Split one labelled CSV file into a training set and a held-out test set.
2. Fit all cleaning and transformation parameters on the training set only.
3. Build one geographically weighted local symbolic-regression task per test
   location.
4. Build geographic-semantic transfer chains from local-neighbour overlap,
   spatial proximity, and weighted feature-distribution similarity.
5. Run a full GWSR search for chain anchors; re-score and transfer the final
   parent formula population to eligible child tasks.
6. Reject unsafe transferred predictions and fall back to a full GWSR search.
7. Write predictions, local formulas, transfer decisions, metrics, and the
   exact run configuration to the output directory.

Example (the original heat-flow column names)
----------------------------------------------
python gst_gwsr_core.py ^
  --input heat_flow_stations.csv ^
  --target heat_flow --lon lon --lat lat ^
  --features topo,bouguer,sediment,moho,lab,cpd,lithos_km,volcano ^
  --output heat_flow_gst_gwsr

For a different data set, replace the file and column names.  If --features is
omitted, every numeric column except the target and coordinate columns is used;
explicitly providing --features is recommended so that numeric ID columns are
not unintentionally treated as predictors.

Dependencies
------------
numpy, pandas, scipy, scikit-learn, joblib, gplearn

The defaults reproduce the *core configuration* of the original safe-transfer
GST-GWSR run: k=25, population=1000, full generations=20, adaptation
generations=5, a Gaussian local kernel, and the original transfer thresholds.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from gplearn.genetic import SymbolicRegressor
except ImportError as exc:  # pragma: no cover - gives the user a useful error
    raise SystemExit(
        "Missing dependency: gplearn. Install dependencies with:\n"
        "  python -m pip install numpy pandas scipy scikit-learn joblib gplearn"
    ) from exc


EPSILON = 1e-8
EARTH_RADIUS_KM = 6371.0088
FUNCTION_SET = ("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg")


@dataclass(frozen=True)
class RunConfig:
    """All values that determine the portable GST-GWSR core run."""

    target: str
    lon: str
    lat: str
    features: tuple[str, ...]
    test_size: float = 0.20
    split_seed: int = 42
    model_seed: int = 42
    iqr_multiplier: float = 3.0
    target_transform: str = "auto"
    distance_metric: str = "euclidean_degrees"
    k_neighbors: int = 25
    population_size: int = 1000
    full_generations: int = 20
    adapt_generations: int = 5
    transfer_chains: int = 24
    n_jobs: int = 1
    neighbor_similarity_weight: float = 0.40
    spatial_similarity_weight: float = 0.30
    feature_similarity_weight: float = 0.30
    spatial_similarity_scale_km: float = 100.0
    min_neighbor_jaccard: float = 0.50
    min_task_similarity: float = 0.60
    guard_z_threshold: float = 2.5
    max_query_points: int = 0


@dataclass
class TransformState:
    """Training-only feature and target transformations."""

    x_scaler: StandardScaler
    y_scaler: StandardScaler
    target_transform: str
    iqr_lower: float
    iqr_upper: float


@dataclass
class LocalTaskCache:
    """Fixed local neighbourhood information for every prediction task."""

    neighbor_indices: np.ndarray
    spatial_weights: np.ndarray
    local_distances: np.ndarray
    descriptors: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portable safe-transfer GST-GWSR for a labelled spatial CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input labelled CSV file.")
    parser.add_argument("--target", required=True, help="Target-variable column name.")
    parser.add_argument("--lon", default="lon", help="Longitude/X-coordinate column name.")
    parser.add_argument("--lat", default="lat", help="Latitude/Y-coordinate column name.")
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated predictor columns. If omitted, infer numeric columns.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output directory. Default: <input_stem>_gst_gwsr_output.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=3.0,
        help="Training-only target IQR cleaning; set 0 to disable.",
    )
    parser.add_argument(
        "--target-transform",
        choices=("auto", "log", "none"),
        default="auto",
        help="'auto' uses log only when all retained training targets are positive.",
    )
    parser.add_argument(
        "--distance-metric",
        choices=("euclidean_degrees", "haversine_km"),
        default="euclidean_degrees",
        help="Local-neighbour distance. The original reported core run used euclidean_degrees.",
    )
    parser.add_argument("--k-neighbors", type=int, default=25)
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--full-generations", type=int, default=20)
    parser.add_argument("--adapt-generations", type=int, default=5)
    parser.add_argument("--transfer-chains", type=int, default=24)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Parallel chains; each symbolic-regression model remains single-threaded.",
    )
    parser.add_argument("--neighbor-similarity-weight", type=float, default=0.40)
    parser.add_argument("--spatial-similarity-weight", type=float, default=0.30)
    parser.add_argument("--feature-similarity-weight", type=float, default=0.30)
    parser.add_argument("--spatial-similarity-scale-km", type=float, default=100.0)
    parser.add_argument("--min-neighbor-jaccard", type=float, default=0.50)
    parser.add_argument("--min-task-similarity", type=float, default=0.60)
    parser.add_argument("--guard-z-threshold", type=float, default=2.5)
    parser.add_argument(
        "--max-query-points",
        type=int,
        default=0,
        help="Optional small smoke-test cap. 0 means use every held-out point.",
    )
    return parser.parse_args()


def validate_config(config: RunConfig) -> None:
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("--test-size must be strictly between 0 and 1.")
    if config.k_neighbors < 5:
        raise ValueError("--k-neighbors must be at least 5.")
    if config.population_size < 10:
        raise ValueError("--population-size must be at least 10.")
    if config.full_generations < 1 or config.adapt_generations < 1:
        raise ValueError("Generation counts must be positive integers.")
    if config.transfer_chains < 1:
        raise ValueError("--transfer-chains must be at least 1.")
    if config.n_jobs == 0:
        raise ValueError("--n-jobs cannot be 0.")
    if config.iqr_multiplier < 0:
        raise ValueError("--iqr-multiplier cannot be negative.")
    if config.spatial_similarity_scale_km <= 0:
        raise ValueError("--spatial-similarity-scale-km must be positive.")
    if config.guard_z_threshold <= 0:
        raise ValueError("--guard-z-threshold must be positive.")
    weights = np.asarray(
        [
            config.neighbor_similarity_weight,
            config.spatial_similarity_weight,
            config.feature_similarity_weight,
        ],
        dtype=float,
    )
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("The three task-similarity weights must be non-negative and sum to 1.")


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
    if not features:
        raise ValueError(
            "No predictor columns were found. Supply --features col_a,col_b,... explicitly."
        )
    if len(set(features)) != len(features):
        raise ValueError("--features contains duplicate column names.")
    return features


def load_and_clean_input(
    input_path: Path,
    target: str,
    lon: str,
    lat: str,
    features: tuple[str, ...],
) -> tuple[pd.DataFrame, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    frame = pd.read_csv(input_path)
    required = [target, lon, lat, *features]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"The input CSV is missing required columns: {missing}")

    data = frame.loc[:, required].copy()
    data.insert(0, "source_row", frame.index.to_numpy())
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = np.isfinite(data[required].to_numpy(dtype=float)).all(axis=1)
    removed = int((~valid).sum())
    data = data.loc[valid].reset_index(drop=True)
    if len(data) < 12:
        raise ValueError(
            "Fewer than 12 complete rows remain after removing missing/non-numeric values."
        )
    return data, removed


def split_and_transform(
    data: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, TransformState, int]:
    """Create a hold-out evaluation and fit all transforms only on training rows."""
    all_positions = np.arange(len(data))
    train_positions, test_positions = train_test_split(
        all_positions,
        test_size=config.test_size,
        random_state=config.split_seed,
    )
    raw_train = data.iloc[train_positions].reset_index(drop=True)
    test = data.iloc[test_positions].reset_index(drop=True)

    target_values = raw_train[config.target].to_numpy(dtype=float)
    if config.iqr_multiplier > 0:
        q1, q3 = np.quantile(target_values, [0.25, 0.75])
        iqr = float(q3 - q1)
        lower = float(q1 - config.iqr_multiplier * iqr)
        upper = float(q3 + config.iqr_multiplier * iqr)
        retain = raw_train[config.target].between(lower, upper).to_numpy()
    else:
        lower, upper = float("-inf"), float("inf")
        retain = np.ones(len(raw_train), dtype=bool)
    train = raw_train.loc[retain].reset_index(drop=True)
    removed_outliers = int((~retain).sum())

    if len(train) < max(10, config.k_neighbors):
        raise ValueError(
            "Too few training rows remain after training-only IQR cleaning for the requested "
            f"k={config.k_neighbors}. Retained training rows: {len(train)}."
        )

    target_transform = config.target_transform
    if target_transform == "auto":
        target_transform = "log" if (train[config.target] > 0).all() else "none"
    if target_transform == "log" and not (train[config.target] > 0).all():
        raise ValueError("--target-transform log requires every retained training target to be > 0.")

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(train.loc[:, config.features].to_numpy(dtype=float))
    x_test = x_scaler.transform(test.loc[:, config.features].to_numpy(dtype=float))

    y_train_raw = train[config.target].to_numpy(dtype=float)
    if target_transform == "log":
        y_train_for_scaling = np.log(y_train_raw)
    else:
        y_train_for_scaling = y_train_raw
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_for_scaling.reshape(-1, 1)).ravel()

    state = TransformState(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        target_transform=target_transform,
        iqr_lower=lower,
        iqr_upper=upper,
    )
    return train, test, x_train, x_test, y_train, state, removed_outliers


def haversine_distance_km(first_lon_lat: np.ndarray, second_lon_lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances for arrays ordered as [longitude, latitude]."""
    first = np.asarray(first_lon_lat, dtype=float)
    second = np.asarray(second_lon_lat, dtype=float)
    lon_a = np.radians(first[:, 0])[:, None]
    lat_a = np.radians(first[:, 1])[:, None]
    lon_b = np.radians(second[:, 0])[None, :]
    lat_b = np.radians(second[:, 1])[None, :]
    delta_lon = lon_a - lon_b
    delta_lat = lat_a - lat_b
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def local_distance_matrix(
    query_coords: np.ndarray,
    train_coords: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "euclidean_degrees":
        return cdist(query_coords, train_coords, metric="euclidean")
    if metric == "haversine_km":
        return haversine_distance_km(query_coords, train_coords)
    raise ValueError(f"Unknown local distance metric: {metric}")


def build_local_task_cache(
    distance_query_train: np.ndarray,
    x_train: np.ndarray,
    k_neighbors: int,
) -> LocalTaskCache:
    """Compute kNN membership, Gaussian weights, and task descriptors once."""
    n_queries, n_train = distance_query_train.shape
    k_use = min(k_neighbors, n_train)
    if k_use < 5:
        raise ValueError("At least five training neighbours are required.")

    neighbor_indices = np.empty((n_queries, k_use), dtype=int)
    local_distances = np.empty((n_queries, k_use), dtype=float)
    spatial_weights = np.empty((n_queries, k_use), dtype=float)
    descriptors = np.empty((n_queries, 2 * x_train.shape[1]), dtype=float)

    for query_id in range(n_queries):
        positions = np.argsort(distance_query_train[query_id])[:k_use]
        distances = distance_query_train[query_id, positions]
        bandwidth = float(distances[-1]) + EPSILON
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights /= max(float(weights.sum()), EPSILON)

        local_x = x_train[positions]
        local_mean = np.average(local_x, axis=0, weights=weights)
        local_var = np.average((local_x - local_mean) ** 2, axis=0, weights=weights)

        neighbor_indices[query_id] = positions
        local_distances[query_id] = distances
        spatial_weights[query_id] = weights
        descriptors[query_id] = np.concatenate(
            [local_mean, np.sqrt(np.maximum(local_var, 0.0))]
        )

    return LocalTaskCache(
        neighbor_indices=neighbor_indices,
        spatial_weights=spatial_weights,
        local_distances=local_distances,
        descriptors=descriptors,
    )


def calculate_jaccard_similarity(neighbor_indices: np.ndarray) -> np.ndarray:
    count = len(neighbor_indices)
    jaccard = np.eye(count, dtype=float)
    sets = [set(row.tolist()) for row in neighbor_indices]
    for i in range(count):
        for j in range(i + 1, count):
            union_size = len(sets[i] | sets[j])
            value = len(sets[i] & sets[j]) / union_size if union_size else 0.0
            jaccard[i, j] = value
            jaccard[j, i] = value
    return jaccard


def calculate_task_similarity(
    query_coords: np.ndarray,
    local_cache: LocalTaskCache,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return total similarity and its geographic, overlap, and feature components."""
    jaccard = calculate_jaccard_similarity(local_cache.neighbor_indices)
    spatial_distance_km = haversine_distance_km(query_coords, query_coords)
    spatial_similarity = np.exp(-spatial_distance_km / config.spatial_similarity_scale_km)

    feature_distance = cdist(local_cache.descriptors, local_cache.descriptors, metric="euclidean")
    positive_distance = feature_distance[feature_distance > 0]
    feature_scale = float(np.median(positive_distance)) if len(positive_distance) else 1.0
    feature_similarity = np.exp(-feature_distance / max(feature_scale, EPSILON))

    total_similarity = (
        config.neighbor_similarity_weight * jaccard
        + config.spatial_similarity_weight * spatial_similarity
        + config.feature_similarity_weight * feature_similarity
    )
    np.fill_diagonal(total_similarity, 1.0)
    return total_similarity, spatial_distance_km, jaccard, feature_similarity


def projected_coordinates_for_clustering(query_coords: np.ndarray) -> np.ndarray:
    """Simple local km projection used only to divide independent transfer chains."""
    mean_lat_rad = np.radians(float(np.mean(query_coords[:, 1])))
    x = query_coords[:, 0] * 111.320 * np.cos(mean_lat_rad)
    y = query_coords[:, 1] * 110.574
    return np.column_stack([x, y])


def build_transfer_chains(
    query_frame: pd.DataFrame,
    task_similarity: np.ndarray,
    spatial_distance_km: np.ndarray,
    jaccard: np.ndarray,
    feature_similarity: np.ndarray,
    config: RunConfig,
) -> pd.DataFrame:
    """Build deterministic anchor-to-child chains without using test target values."""
    count = len(query_frame)
    n_chains = min(config.transfer_chains, count)
    cluster_ids = KMeans(
        n_clusters=n_chains,
        random_state=config.split_seed,
        n_init=20,
    ).fit_predict(projected_coordinates_for_clustering(query_frame[[config.lon, config.lat]].to_numpy()))

    rows: list[dict[str, Any]] = []
    for chain_id in range(n_chains):
        members = np.flatnonzero(cluster_ids == chain_id).tolist()
        if not members:
            continue
        if len(members) == 1:
            anchor = members[0]
        else:
            within = task_similarity[np.ix_(members, members)].copy()
            np.fill_diagonal(within, 0.0)
            anchor = members[int(np.argmax(within.mean(axis=1)))]

        processed = [anchor]
        unprocessed = set(members) - {anchor}
        rows.append(
            {
                "chain_id": chain_id,
                "order_in_chain": 0,
                "query_id": anchor,
                "source_row": int(query_frame.iloc[anchor]["source_row"]),
                "is_anchor": True,
                "parent_query_id": -1,
                "parent_distance_km": np.nan,
                "parent_neighbor_jaccard": np.nan,
                "parent_feature_similarity": np.nan,
                "parent_task_similarity": np.nan,
                "recommended_transfer": False,
            }
        )

        order = 1
        while unprocessed:
            best_child: Optional[int] = None
            best_parent: Optional[int] = None
            best_score = -np.inf
            for child in sorted(unprocessed):
                scores = task_similarity[child, processed]
                parent_position = int(np.argmax(scores))
                candidate_parent = processed[parent_position]
                score = float(scores[parent_position])
                if score > best_score:
                    best_child = child
                    best_parent = candidate_parent
                    best_score = score
            assert best_child is not None and best_parent is not None
            transfer = bool(
                jaccard[best_child, best_parent] >= config.min_neighbor_jaccard
                and task_similarity[best_child, best_parent] >= config.min_task_similarity
            )
            rows.append(
                {
                    "chain_id": chain_id,
                    "order_in_chain": order,
                    "query_id": best_child,
                    "source_row": int(query_frame.iloc[best_child]["source_row"]),
                    "is_anchor": False,
                    "parent_query_id": best_parent,
                    "parent_distance_km": float(spatial_distance_km[best_child, best_parent]),
                    "parent_neighbor_jaccard": float(jaccard[best_child, best_parent]),
                    "parent_feature_similarity": float(feature_similarity[best_child, best_parent]),
                    "parent_task_similarity": float(task_similarity[best_child, best_parent]),
                    "recommended_transfer": transfer,
                }
            )
            processed.append(best_child)
            unprocessed.remove(best_child)
            order += 1

    return pd.DataFrame(rows).sort_values(["chain_id", "order_in_chain"]).reset_index(drop=True)


def build_symbolic_regressor(config: RunConfig, generations: int, warm_start: bool) -> SymbolicRegressor:
    """The GWSR genetic-programming configuration from the original core run."""
    return SymbolicRegressor(
        population_size=config.population_size,
        generations=generations,
        tournament_size=20,
        init_depth=(2, 5),
        init_method="half and half",
        function_set=list(FUNCTION_SET),
        metric="mse",
        parsimony_coefficient=0.01,
        const_range=(-1.0, 1.0),
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.8,
        feature_names=list(config.features),
        warm_start=warm_start,
        low_memory=True,
        n_jobs=1,
        verbose=0,
        random_state=config.model_seed,
    )


def fit_full_gwsr(
    x_local: np.ndarray,
    y_local: np.ndarray,
    weights: np.ndarray,
    config: RunConfig,
    query_id: int,
) -> SymbolicRegressor:
    model = build_symbolic_regressor(config, config.full_generations, warm_start=False)
    model.set_params(random_state=config.model_seed + query_id)
    model.fit(x_local, y_local, sample_weight=weights)
    return model


def empty_run_details() -> dict[str, list[Any]]:
    return {
        "generation": [],
        "average_length": [],
        "average_fitness": [],
        "best_length": [],
        "best_fitness": [],
        "best_oob_fitness": [],
        "generation_time": [],
    }


def rescore_parent_population(
    parent_population: list[Any],
    x_local: np.ndarray,
    y_local: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[Any], dict[str, float]]:
    """Re-evaluate inherited formulas using only the child's weighted training task."""
    population = copy.deepcopy(parent_population)
    fitness_values: list[float] = []
    for program in population:
        if program is None:
            raise ValueError("Inherited final population contains an empty program.")
        raw_fitness = float(program.raw_fitness(x_local, y_local, weights))
        if not np.isfinite(raw_fitness):
            raw_fitness = float(np.finfo(float).max / 1000.0)
        program.raw_fitness_ = raw_fitness
        program.fitness_ = float(program.fitness())
        program.oob_fitness_ = np.nan
        fitness_values.append(raw_fitness)
    values = np.asarray(fitness_values, dtype=float)
    return population, {
        "rescored_parent_best_mse": float(np.min(values)),
        "rescored_parent_median_mse": float(np.median(values)),
        "rescored_parent_mean_mse": float(np.mean(values)),
    }


def fit_rescored_transfer(
    parent_population: list[Any],
    x_local: np.ndarray,
    y_local: np.ndarray,
    weights: np.ndarray,
    config: RunConfig,
    query_id: int,
) -> tuple[SymbolicRegressor, dict[str, float]]:
    population, diagnostics = rescore_parent_population(
        parent_population, x_local, y_local, weights
    )
    # gplearn's warm_start continues after its existing first population.  A
    # parent final population is injected as that first population, and only
    # the configured number of child adaptation generations is executed.
    model = build_symbolic_regressor(
        config,
        generations=1 + config.adapt_generations,
        warm_start=True,
    )
    model.set_params(random_state=config.model_seed + query_id)
    model._programs = [population]
    model.run_details_ = empty_run_details()
    model.fit(x_local, y_local, sample_weight=weights)
    return model, diagnostics


def finite_prediction_or_local_mean(
    model: SymbolicRegressor,
    x_query: np.ndarray,
    y_local: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, str, bool]:
    prediction = float(model.predict(x_query.reshape(1, -1))[0])
    expression = str(model._program)
    clipped = False
    # The threshold is in standardized target space and matches the original
    # implementation. It prevents invalid protected-operation outputs from
    # becoming a reported prediction.
    if not np.isfinite(prediction) or abs(prediction) >= 5.0:
        prediction = float(np.average(y_local, weights=weights))
        expression = f"CLIPPED({expression[:180]})"
        clipped = True
    return prediction, expression, clipped


def evaluate_local_model(
    model: SymbolicRegressor,
    x_local: np.ndarray,
    y_local: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    local_prediction = np.asarray(model.predict(x_local), dtype=float)
    if not np.isfinite(local_prediction).all():
        return np.nan, np.nan
    local_r2 = float(r2_score(y_local, local_prediction))
    local_weighted_mse = float(
        np.average((y_local - local_prediction) ** 2, weights=weights)
    )
    return local_r2, local_weighted_mse


def local_z_score(
    prediction: float,
    y_local: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    mean = float(np.average(y_local, weights=weights))
    variance = float(np.average((y_local - mean) ** 2, weights=weights))
    std = max(float(np.sqrt(max(variance, 0.0))), EPSILON)
    return float((prediction - mean) / std), mean, std


def local_task_arrays(
    query_id: int,
    local_cache: LocalTaskCache,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = local_cache.neighbor_indices[query_id]
    return x_train[positions], y_train[positions], local_cache.spatial_weights[query_id], positions


def predict_chain(
    chain_id: int,
    chain_rows: pd.DataFrame,
    local_cache: LocalTaskCache,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    config: RunConfig,
) -> list[dict[str, Any]]:
    """Run one chain sequentially; separate chains can safely run in parallel."""
    del chain_id  # The value is retained in chain_rows and only helps joblib grouping.
    ordered = chain_rows.sort_values("order_in_chain").reset_index(drop=True)
    child_counts = ordered.loc[ordered["parent_query_id"] >= 0, "parent_query_id"].value_counts().to_dict()
    populations: dict[int, Optional[list[Any]]] = {}
    records: list[dict[str, Any]] = []

    for row in ordered.itertuples(index=False):
        started = time.perf_counter()
        query_id = int(row.query_id)
        parent_id = int(row.parent_query_id)
        x_local, y_local, weights, positions = local_task_arrays(
            query_id, local_cache, x_train, y_train
        )

        if bool(row.is_anchor):
            requested = "anchor_full_search"
        elif bool(row.recommended_transfer):
            requested = "rescored_transfer"
        else:
            requested = "independent_full_search"

        strategy = requested
        fallback_reason = ""
        error_message = ""
        warm_attempted = False
        guard_triggered = False
        rescoring = {
            "rescored_parent_best_mse": np.nan,
            "rescored_parent_median_mse": np.nan,
            "rescored_parent_mean_mse": np.nan,
        }

        candidate_prediction = np.nan
        candidate_expression = ""
        candidate_z = np.nan
        candidate_clipped = False
        model: Optional[SymbolicRegressor] = None
        population: Optional[list[Any]] = None

        try:
            if requested == "rescored_transfer":
                parent_population = populations.get(parent_id)
                if parent_population is None:
                    strategy = "fallback_full_search"
                    fallback_reason = "parent population unavailable"
                    model = fit_full_gwsr(x_local, y_local, weights, config, query_id)
                else:
                    warm_attempted = True
                    candidate_model, rescoring = fit_rescored_transfer(
                        parent_population,
                        x_local,
                        y_local,
                        weights,
                        config,
                        query_id,
                    )
                    candidate_prediction, candidate_expression, candidate_clipped = finite_prediction_or_local_mean(
                        candidate_model, x_query[query_id], y_local, weights
                    )
                    candidate_z, _, _ = local_z_score(candidate_prediction, y_local, weights)
                    unsafe = bool(
                        candidate_clipped
                        or not np.isfinite(candidate_z)
                        or abs(candidate_z) > config.guard_z_threshold
                    )
                    if unsafe:
                        guard_triggered = True
                        strategy = "guard_full_search"
                        fallback_reason = (
                            "unsafe transferred prediction: "
                            f"local_standardized_z={candidate_z:.6f}"
                        )
                        model = fit_full_gwsr(x_local, y_local, weights, config, query_id)
                    else:
                        strategy = "rescored_transfer_accepted"
                        model = candidate_model
            else:
                model = fit_full_gwsr(x_local, y_local, weights, config, query_id)

            assert model is not None
            prediction, expression, clipped = finite_prediction_or_local_mean(
                model, x_query[query_id], y_local, weights
            )
            local_fit_r2, local_weighted_mse = evaluate_local_model(
                model, x_local, y_local, weights
            )
            final_z, local_mean, local_std = local_z_score(prediction, y_local, weights)
            population = model._programs[-1]
            success = True
        except Exception as exc:
            # Any transfer-side error should not terminate a formal run. It
            # gets one independent full-search attempt, then a transparent
            # local-mean output if the model itself cannot be fitted.
            if requested == "rescored_transfer" and warm_attempted:
                try:
                    strategy = "transfer_error_full_search"
                    fallback_reason = f"transfer error: {str(exc)[:180]}"
                    model = fit_full_gwsr(x_local, y_local, weights, config, query_id)
                    prediction, expression, clipped = finite_prediction_or_local_mean(
                        model, x_query[query_id], y_local, weights
                    )
                    local_fit_r2, local_weighted_mse = evaluate_local_model(
                        model, x_local, y_local, weights
                    )
                    final_z, local_mean, local_std = local_z_score(prediction, y_local, weights)
                    population = model._programs[-1]
                    success = True
                except Exception as second_exc:
                    error_message = f"transfer error: {exc}; full-search error: {second_exc}"
                    prediction = float(np.average(y_local, weights=weights))
                    expression = "FAILED_LOCAL_WEIGHTED_MEAN"
                    clipped = False
                    local_fit_r2 = np.nan
                    local_weighted_mse = float(np.average((y_local - prediction) ** 2, weights=weights))
                    final_z, local_mean, local_std = local_z_score(prediction, y_local, weights)
                    success = False
            else:
                error_message = str(exc)
                prediction = float(np.average(y_local, weights=weights))
                expression = "FAILED_LOCAL_WEIGHTED_MEAN"
                clipped = False
                local_fit_r2 = np.nan
                local_weighted_mse = float(np.average((y_local - prediction) ** 2, weights=weights))
                final_z, local_mean, local_std = local_z_score(prediction, y_local, weights)
                success = False

        # Retain only populations that a later child may need, then release
        # them once every recorded child in this chain has been processed.
        if child_counts.get(query_id, 0) > 0:
            populations[query_id] = population
        if parent_id >= 0:
            child_counts[parent_id] = child_counts.get(parent_id, 0) - 1
            if child_counts[parent_id] <= 0:
                populations.pop(parent_id, None)

        records.append(
            {
                "chain_id": int(row.chain_id),
                "order_in_chain": int(row.order_in_chain),
                "query_id": query_id,
                "source_row": int(row.source_row),
                "is_anchor": bool(row.is_anchor),
                "parent_query_id": parent_id,
                "requested_strategy": requested,
                "final_strategy": strategy,
                "recommended_transfer": bool(row.recommended_transfer),
                "warm_transfer_attempted": warm_attempted,
                "transfer_guard_triggered": guard_triggered,
                "fallback_reason": fallback_reason,
                "failure_reason": error_message,
                "success": success,
                "prediction_scaled": prediction,
                "expression": expression,
                "prediction_clipped": clipped,
                "local_fit_r2": local_fit_r2,
                "local_weighted_mse": local_weighted_mse,
                "prediction_local_z": final_z,
                "local_target_mean_scaled": local_mean,
                "local_target_std_scaled": local_std,
                "candidate_prediction_scaled": candidate_prediction,
                "candidate_expression": candidate_expression,
                "candidate_prediction_local_z": candidate_z,
                "candidate_prediction_clipped": candidate_clipped,
                "rescored_parent_best_mse": rescoring["rescored_parent_best_mse"],
                "rescored_parent_median_mse": rescoring["rescored_parent_median_mse"],
                "rescored_parent_mean_mse": rescoring["rescored_parent_mean_mse"],
                "local_neighbor_count": int(len(positions)),
                "local_bandwidth": float(local_cache.local_distances[query_id, -1]),
                "local_effective_sample_size": float(1.0 / np.sum(weights**2)),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    return records


def inverse_target(values: np.ndarray, state: TransformState) -> np.ndarray:
    restored = state.y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()
    return np.exp(np.clip(restored, -30.0, 30.0)) if state.target_transform == "log" else restored


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE_percent": np.nan}
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    nonzero = np.abs(y_true) > EPSILON
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if np.any(nonzero) else np.nan
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_percent": mape,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def run(config: RunConfig, input_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    input_frame = pd.read_csv(input_path)
    data, removed_missing = load_and_clean_input(
        input_path, config.target, config.lon, config.lat, config.features
    )
    train, test, x_train, x_test, y_train, transform_state, removed_outliers = split_and_transform(
        data, config
    )

    if config.max_query_points > 0:
        test = test.iloc[: config.max_query_points].reset_index(drop=True)
        x_test = x_test[: config.max_query_points]
    if len(test) < 2:
        raise ValueError("At least two held-out query points are required.")

    train_coords = train[[config.lon, config.lat]].to_numpy(dtype=float)
    query_coords = test[[config.lon, config.lat]].to_numpy(dtype=float)
    distance_query_train = local_distance_matrix(query_coords, train_coords, config.distance_metric)
    local_cache = build_local_task_cache(distance_query_train, x_train, config.k_neighbors)
    task_similarity, spatial_distance_km, jaccard, feature_similarity = calculate_task_similarity(
        query_coords, local_cache, config
    )
    chains = build_transfer_chains(
        test,
        task_similarity,
        spatial_distance_km,
        jaccard,
        feature_similarity,
        config,
    )

    grouped_chains = [
        (int(chain_id), group.copy())
        for chain_id, group in chains.groupby("chain_id", sort=True)
    ]
    if config.n_jobs == 1 or len(grouped_chains) == 1:
        results_by_chain = [
            predict_chain(chain_id, chain, local_cache, x_train, y_train, x_test, config)
            for chain_id, chain in grouped_chains
        ]
    else:
        results_by_chain = Parallel(n_jobs=config.n_jobs, verbose=5)(
            delayed(predict_chain)(
                chain_id, chain, local_cache, x_train, y_train, x_test, config
            )
            for chain_id, chain in grouped_chains
        )

    result_rows = [row for chain_rows in results_by_chain for row in chain_rows]
    predictions = pd.DataFrame(result_rows).sort_values("query_id").reset_index(drop=True)
    predictions.insert(1, "lon", test[config.lon].to_numpy(dtype=float))
    predictions.insert(2, "lat", test[config.lat].to_numpy(dtype=float))
    predictions["observed_target"] = test[config.target].to_numpy(dtype=float)
    predictions["prediction"] = inverse_target(
        predictions["prediction_scaled"].to_numpy(dtype=float), transform_state
    )
    predictions["residual"] = predictions["prediction"] - predictions["observed_target"]

    metrics = metric_summary(
        predictions["observed_target"].to_numpy(dtype=float),
        predictions["prediction"].to_numpy(dtype=float),
    )
    elapsed = float(time.perf_counter() - started)
    strategy_counts = predictions["final_strategy"].value_counts().to_dict()

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "gst_gwsr_predictions_and_formulas.csv", index=False, encoding="utf-8-sig")
    chains.to_csv(output_dir / "gst_gwsr_transfer_chains.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "query_id": np.arange(len(test)),
            "source_row": test["source_row"].to_numpy(),
            "lon": query_coords[:, 0],
            "lat": query_coords[:, 1],
            "neighbor_indices_in_clean_train": [
                ";".join(map(str, row)) for row in local_cache.neighbor_indices
            ],
            "local_bandwidth": local_cache.local_distances[:, -1],
            "local_effective_sample_size": 1.0 / np.sum(local_cache.spatial_weights**2, axis=1),
        }
    ).to_csv(output_dir / "gst_gwsr_local_tasks.csv", index=False, encoding="utf-8-sig")

    summary = {
        "method": "Geographic-Semantic Transfer enhanced Geographically Weighted Symbolic Regression (GST-GWSR)",
        "input_csv": str(input_path.resolve()),
        "input_rows": int(len(input_frame)),
        "rows_removed_missing_or_non_numeric": removed_missing,
        "raw_training_rows": int(len(train) + removed_outliers),
        "clean_training_rows": int(len(train)),
        "training_outliers_removed_by_iqr": removed_outliers,
        "held_out_query_rows": int(len(test)),
        "target_transform_used": transform_state.target_transform,
        "training_iqr_lower": transform_state.iqr_lower,
        "training_iqr_upper": transform_state.iqr_upper,
        "metrics_on_held_out_targets": metrics,
        "strategy_counts": strategy_counts,
        "transfer_recommended_count": int(chains["recommended_transfer"].sum()),
        "transfer_accepted_count": int((predictions["final_strategy"] == "rescored_transfer_accepted").sum()),
        "safe_fallback_count": int((predictions["final_strategy"] == "guard_full_search").sum()),
        "run_elapsed_seconds": elapsed,
        "config": asdict(config),
        "notes": [
            "All preprocessing parameters are fit using training rows only.",
            "Held-out target values are not used for neighbourhood construction, task similarity, transfer decisions, formula search, or safety gates.",
            "The local-distance default is euclidean_degrees to match the original reported core run. Use haversine_km for a new geographically broad data set if that is your predefined methodological choice.",
        ],
    }
    with (output_dir / "gst_gwsr_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    raw_frame = pd.read_csv(args.input)
    features = resolve_features(raw_frame, args)
    config = RunConfig(
        target=args.target,
        lon=args.lon,
        lat=args.lat,
        features=features,
        test_size=args.test_size,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        iqr_multiplier=args.iqr_multiplier,
        target_transform=args.target_transform,
        distance_metric=args.distance_metric,
        k_neighbors=args.k_neighbors,
        population_size=args.population_size,
        full_generations=args.full_generations,
        adapt_generations=args.adapt_generations,
        transfer_chains=args.transfer_chains,
        n_jobs=args.n_jobs,
        neighbor_similarity_weight=args.neighbor_similarity_weight,
        spatial_similarity_weight=args.spatial_similarity_weight,
        feature_similarity_weight=args.feature_similarity_weight,
        spatial_similarity_scale_km=args.spatial_similarity_scale_km,
        min_neighbor_jaccard=args.min_neighbor_jaccard,
        min_task_similarity=args.min_task_similarity,
        guard_z_threshold=args.guard_z_threshold,
        max_query_points=args.max_query_points,
    )
    validate_config(config)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_gst_gwsr_output")
    )

    print("=" * 84)
    print("Portable GST-GWSR: safe geographic-semantic transfer for local symbolic regression")
    print("=" * 84)
    print(f"Input: {input_path}")
    print(f"Target / coordinates: {config.target} / ({config.lon}, {config.lat})")
    print(f"Features ({len(config.features)}): {', '.join(config.features)}")
    print(f"Output: {output_dir}")
    print(
        "Core configuration: "
        f"k={config.k_neighbors}, population={config.population_size}, "
        f"full generations={config.full_generations}, adaptation={config.adapt_generations}"
    )
    print("Test target values used during search or transfer: NO")

    summary = run(config, input_path, output_dir)
    metrics = summary["metrics_on_held_out_targets"]
    print("\nRun complete")
    print(
        "  Held-out GST-GWSR: "
        f"R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  "
        f"MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE_percent']:.2f}%"
    )
    print(f"  Strategy counts: {summary['strategy_counts']}")
    print(f"  Output folder: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nGST-GWSR stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
