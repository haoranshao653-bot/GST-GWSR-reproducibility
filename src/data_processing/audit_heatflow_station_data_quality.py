"""Audit spatial sampling and target-quality signals in the heat-flow station table.

This script is diagnostic only: it does not remove stations or train a model.
It reveals duplicated/near-duplicated locations, spatial clustering, random-split
proximity, and global/local heat-flow outlier candidates.
"""

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


DATA_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju")
INPUT_FILE = DATA_DIR / "active_fault_station_features" / "heat_flow_stations_with_fault_distance.csv"
OUTPUT_DIR = DATA_DIR / "station_data_quality_audit"
LON = "lon"
LAT = "lat"
TARGET = "heat_flow"
SPLIT_SEED = 42
TEST_SIZE = 0.2
LOCAL_NEIGHBOR_K = 20
NEAR_DUPLICATE_KM = 1.0


def local_xy_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Local equirectangular coordinates, adequate for station-distance diagnostics."""
    radius_km = 6371.0088
    lat0 = np.deg2rad(np.mean(lat))
    x = radius_km * np.deg2rad(lon) * np.cos(lat0)
    y = radius_km * np.deg2rad(lat)
    return np.column_stack([x, y])


def robust_limits(values: pd.Series) -> tuple[float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    low, high = np.percentile(clean, [2, 98])
    return float(low), float(high if high > low else low + 1.0)


def save_audit_figure(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    axis = axes[0]
    low, high = robust_limits(frame["nearest_station_km"])
    points = axis.scatter(
        frame[LON], frame[LAT], c=frame["nearest_station_km"], s=20,
        cmap="viridis_r", vmin=low, vmax=high, edgecolors="none",
    )
    fig.colorbar(points, ax=axis, fraction=0.046, pad=0.04, label="最近邻站距离 (km)")
    axis.set_title("站点空间采样密度")
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.grid(alpha=0.25)

    axis = axes[1]
    low, high = robust_limits(frame[TARGET])
    points = axis.scatter(
        frame[LON], frame[LAT], c=frame[TARGET], s=20,
        cmap="plasma", vmin=low, vmax=high, edgecolors="none",
    )
    fig.colorbar(points, ax=axis, fraction=0.046, pad=0.04, label="热流 (mW/m²)")
    axis.set_title("热流空间分布")
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.grid(alpha=0.25)

    axis = axes[2]
    axis.hist(frame["nearest_station_km"], bins=40, color="#2878b5", edgecolor="white")
    axis.axvline(NEAR_DUPLICATE_KM, color="#d55e00", linestyle="--", label=f"{NEAR_DUPLICATE_KM:g} km")
    axis.set_title("最近邻站距离分布")
    axis.set_xlabel("最近邻站距离 (km)")
    axis.set_ylabel("站点数")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[3]
    low, high = robust_limits(frame["local_heatflow_z"])
    points = axis.scatter(
        frame[LON], frame[LAT], c=frame["local_heatflow_z"], s=20,
        cmap="coolwarm", vmin=low, vmax=high, edgecolors="none",
    )
    fig.colorbar(points, ax=axis, fraction=0.046, pad=0.04, label="局部热流 z 值")
    axis.set_title(f"局部热流异常诊断（{LOCAL_NEIGHBOR_K} 个邻站）")
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input table not found: {INPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    missing = {LON, LAT, TARGET} - set(df.columns)
    if missing:
        raise KeyError(f"Input table is missing columns: {sorted(missing)}")
    if df[[LON, LAT, TARGET]].isna().any().any():
        raise ValueError("Coordinate or heat-flow columns contain missing values.")

    xy = local_xy_km(df[LON].to_numpy(), df[LAT].to_numpy())
    tree = cKDTree(xy)
    distances, indices = tree.query(xy, k=LOCAL_NEIGHBOR_K + 1)
    df["nearest_station_km"] = distances[:, 1]
    df["nearest_station_index"] = indices[:, 1]
    df["near_duplicate_within_1km"] = df["nearest_station_km"] <= NEAR_DUPLICATE_KM

    neighbor_indices = indices[:, 1:]
    neighbor_q = df[TARGET].to_numpy()[neighbor_indices]
    local_mean = neighbor_q.mean(axis=1)
    local_std = neighbor_q.std(axis=1, ddof=1)
    df["local_neighbor_heatflow_mean"] = local_mean
    df["local_neighbor_heatflow_std"] = local_std
    df["local_heatflow_z"] = (df[TARGET].to_numpy() - local_mean) / np.where(local_std > 1e-9, local_std, np.nan)
    df["local_high_outlier_z3"] = df["local_heatflow_z"] > 3.0

    q1 = float(df[TARGET].quantile(0.25))
    q3 = float(df[TARGET].quantile(0.75))
    iqr = q3 - q1
    global_upper_3iqr = q3 + 3.0 * iqr
    df["global_high_outlier_3iqr"] = df[TARGET] > global_upper_3iqr

    all_indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(all_indices, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    train_tree = cKDTree(xy[train_indices])
    test_to_train_km, nearest_train_pos = train_tree.query(xy[test_indices], k=1)
    split_df = df.iloc[test_indices][[LON, LAT, TARGET]].copy()
    split_df["nearest_train_station_km"] = test_to_train_km
    split_df["nearest_train_original_index"] = train_indices[nearest_train_pos]

    quality_keywords = re.compile(r"quality|uncert|method|probe|source|reference|survey|测量|误差|质量|方法", re.IGNORECASE)
    derived_prefixes = ("gebco_", "emag2_", "dist_to_", "near_active_", "log1p_", "fault_")
    possible_quality_columns = [
        column for column in df.columns
        if quality_keywords.search(column) and not column.lower().startswith(derived_prefixes)
    ]

    near_duplicate_df = df[df["near_duplicate_within_1km"]].sort_values("nearest_station_km")
    local_outlier_df = df[df["local_high_outlier_z3"]].sort_values("local_heatflow_z", ascending=False)
    global_outlier_df = df[df["global_high_outlier_3iqr"]].sort_values(TARGET, ascending=False)
    density_df = (
        df.assign(lon_cell=df[LON].round(1), lat_cell=df[LAT].round(1))
        .groupby(["lon_cell", "lat_cell"], as_index=False)
        .agg(station_count=(TARGET, "size"), heat_flow_mean=(TARGET, "mean"), heat_flow_std=(TARGET, "std"))
        .sort_values("station_count", ascending=False)
    )

    audit_table_path = OUTPUT_DIR / "station_audit_01_all_stations.csv"
    duplicates_path = OUTPUT_DIR / "station_audit_02_near_duplicates_within_1km.csv"
    local_outlier_path = OUTPUT_DIR / "station_audit_03_local_high_outlier_candidates.csv"
    global_outlier_path = OUTPUT_DIR / "station_audit_04_global_high_outlier_3iqr.csv"
    split_path = OUTPUT_DIR / "station_audit_05_test_to_train_nearest_distance.csv"
    density_path = OUTPUT_DIR / "station_audit_06_spatial_density_0p1deg.csv"
    figure_path = OUTPUT_DIR / "station_audit_07_spatial_quality_overview.png"
    summary_path = OUTPUT_DIR / "station_audit_08_summary.json"

    df.to_csv(audit_table_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    near_duplicate_df.to_csv(duplicates_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    local_outlier_df.to_csv(local_outlier_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    global_outlier_df.to_csv(global_outlier_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    density_df.to_csv(density_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    save_audit_figure(df, figure_path)

    summary = {
        "input_file": str(INPUT_FILE),
        "station_count": int(len(df)),
        "spatial_sampling": {
            "nearest_station_km_median": float(df["nearest_station_km"].median()),
            "nearest_station_km_p10": float(df["nearest_station_km"].quantile(0.10)),
            "nearest_station_km_p90": float(df["nearest_station_km"].quantile(0.90)),
            "near_duplicate_within_1km_count": int(df["near_duplicate_within_1km"].sum()),
        },
        "random_split_proximity": {
            "split_seed": SPLIT_SEED,
            "test_count": int(len(split_df)),
            "test_to_train_nearest_km_median": float(split_df["nearest_train_station_km"].median()),
            "test_to_train_nearest_km_p10": float(split_df["nearest_train_station_km"].quantile(0.10)),
            "test_to_train_within_5km_count": int((split_df["nearest_train_station_km"] <= 5).sum()),
            "note": "Very short test-to-train distances indicate interpolation-style evaluation, not independent spatial extrapolation.",
        },
        "heat_flow_outlier_diagnostics": {
            "global_3iqr_upper_bound": global_upper_3iqr,
            "global_high_outlier_3iqr_count": int(df["global_high_outlier_3iqr"].sum()),
            "local_high_outlier_z3_count": int(df["local_high_outlier_z3"].sum()),
        },
        "possible_measurement_quality_columns": possible_quality_columns,
        "note": "The CSV cannot reveal unrecorded field measurement method or uncertainty. Obtain those from original station metadata if available.",
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("Heat-flow station data and spatial-sampling audit")
    print("=" * 76)
    print(f"Stations: {len(df)}")
    print(f"Nearest-station distance: median={df['nearest_station_km'].median():.2f} km")
    print(f"Near duplicates within {NEAR_DUPLICATE_KM:g} km: {int(df['near_duplicate_within_1km'].sum())}")
    print(f"Test-to-train nearest distance: median={split_df['nearest_train_station_km'].median():.2f} km")
    print(f"Test points within 5 km of training: {int((split_df['nearest_train_station_km'] <= 5).sum())}/{len(split_df)}")
    print(f"Global high outliers (3*IQR): {int(df['global_high_outlier_3iqr'].sum())}")
    print(f"Local high outlier candidates (z>3): {int(df['local_high_outlier_z3'].sum())}")
    print(f"Possible measurement-quality columns: {possible_quality_columns or 'none found'}")
    print("\nSaved:")
    for path in [audit_table_path, duplicates_path, local_outlier_path, global_outlier_path, split_path, density_path, figure_path, summary_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
