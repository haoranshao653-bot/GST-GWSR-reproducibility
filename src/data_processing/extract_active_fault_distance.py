"""Calculate distance from each heat-flow station to the nearest GEM active fault.

The output contains one new feature: dist_to_active_fault_km.
Distances are calculated after projecting the South China Sea region to a local
azimuthal-equidistant coordinate system, so the unit is approximately metres.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as error:
    raise SystemExit(
        "Missing dependency: geopandas. Install it with:\n"
        "  pip install geopandas\n"
        f"Details: {error}"
    )

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


DATA_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju")
FAULT_REPOSITORY_DIR = DATA_DIR / "gem-global-active-faults-master"
INPUT_FILE = DATA_DIR / "emag2_station_features" / "heat_flow_stations_with_gebco_and_emag2.csv"
OUTPUT_DIR = DATA_DIR / "active_fault_station_features"
LON_COLUMN = "lon"
LAT_COLUMN = "lat"
TARGET = "heat_flow"

# A local distance-preserving projection centered on the South China Sea.
LOCAL_CRS = "+proj=aeqd +lat_0=12 +lon_0=113 +datum=WGS84 +units=m +no_defs"
STUDY_BUFFER_DEGREES = 5.0


def find_fault_geojson(repository_dir: Path) -> Path:
    if not repository_dir.exists():
        raise FileNotFoundError(f"Fault repository folder not found: {repository_dir}")
    candidates = sorted(repository_dir.rglob("*.geojson"))
    if not candidates:
        raise FileNotFoundError(
            f"No GeoJSON file found below: {repository_dir}\n"
            "Confirm that the GitHub ZIP was fully extracted."
        )
    harmonized = [path for path in candidates if "harmonized" in path.name.lower()]
    return harmonized[0] if harmonized else candidates[0]


def save_map(stations: gpd.GeoDataFrame, faults: gpd.GeoDataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 8))
    faults.plot(ax=axis, color="#4d4d4d", linewidth=0.55, alpha=0.8, label="活动断层")
    low, high = np.percentile(stations["dist_to_active_fault_km"], [2, 98])
    scatter = axis.scatter(
        stations[LON_COLUMN], stations[LAT_COLUMN], c=stations["dist_to_active_fault_km"],
        s=22, cmap="viridis_r", vmin=low, vmax=high, edgecolors="none", label="热流站",
    )
    colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("距最近活动断层 (km)")
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.set_title("南海热流站与 GEM 活动断层距离")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Station table not found: {INPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fault_file = find_fault_geojson(FAULT_REPOSITORY_DIR)
    station_df = pd.read_csv(INPUT_FILE)
    missing = {LON_COLUMN, LAT_COLUMN, TARGET} - set(station_df.columns)
    if missing:
        raise KeyError(f"Input table is missing columns: {sorted(missing)}")

    print("=" * 72)
    print("GEM active-fault distance extraction")
    print("=" * 72)
    print(f"Station table: {INPUT_FILE}")
    print(f"Fault GeoJSON: {fault_file}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Station count: {len(station_df)}")

    station_gdf = gpd.GeoDataFrame(
        station_df.copy(),
        geometry=gpd.points_from_xy(station_df[LON_COLUMN], station_df[LAT_COLUMN]),
        crs="EPSG:4326",
    )
    faults = gpd.read_file(fault_file)
    if faults.crs is None:
        faults = faults.set_crs("EPSG:4326")
    else:
        faults = faults.to_crs("EPSG:4326")
    faults = faults[faults.geometry.notna() & ~faults.geometry.is_empty].copy()

    min_lon = station_df[LON_COLUMN].min() - STUDY_BUFFER_DEGREES
    max_lon = station_df[LON_COLUMN].max() + STUDY_BUFFER_DEGREES
    min_lat = station_df[LAT_COLUMN].min() - STUDY_BUFFER_DEGREES
    max_lat = station_df[LAT_COLUMN].max() + STUDY_BUFFER_DEGREES
    regional_faults = faults.cx[min_lon:max_lon, min_lat:max_lat].copy()
    if regional_faults.empty:
        raise RuntimeError("No GEM faults found near the station region. Check the GeoJSON coordinate system.")
    print(f"Fault traces: global={len(faults)}, regional={len(regional_faults)}")

    # Use every global fault when finding the nearest trace. The regional subset
    # is only for a readable map, not for the distance calculation.
    projected_stations = station_gdf.to_crs(LOCAL_CRS).copy()
    projected_stations["station_id"] = np.arange(len(projected_stations))
    projected_faults_all = faults.to_crs(LOCAL_CRS)
    try:
        nearest = gpd.sjoin_nearest(
            projected_stations[["station_id", "geometry"]],
            projected_faults_all[["geometry"]],
            how="left",
            distance_col="nearest_distance_m",
        )
        distances_m = (
            nearest.groupby("station_id")["nearest_distance_m"]
            .min()
            .reindex(np.arange(len(station_df)))
            .to_numpy()
        )
    except (AttributeError, NotImplementedError):
        # Compatibility fallback for older GeoPandas/Shapely installations.
        try:
            global_fault_union = projected_faults_all.geometry.union_all()
        except AttributeError:
            global_fault_union = projected_faults_all.geometry.unary_union
        distances_m = projected_stations.geometry.distance(global_fault_union).to_numpy()

    if np.isnan(distances_m).any():
        raise RuntimeError("Could not calculate a nearest-fault distance for every station.")
    distances_km = distances_m / 1000.0
    station_df["dist_to_active_fault_km"] = distances_km
    station_df["near_active_fault_50km"] = station_df["dist_to_active_fault_km"] <= 50.0
    station_df["near_active_fault_100km"] = station_df["dist_to_active_fault_km"] <= 100.0

    feature_path = OUTPUT_DIR / "active_fault_01_station_distance_feature.csv"
    combined_path = OUTPUT_DIR / "heat_flow_stations_with_fault_distance.csv"
    map_path = OUTPUT_DIR / "active_fault_02_distance_map.png"
    summary_path = OUTPUT_DIR / "active_fault_03_summary.json"
    station_df[[LON_COLUMN, LAT_COLUMN, "dist_to_active_fault_km", "near_active_fault_50km", "near_active_fault_100km"]].to_csv(
        feature_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    station_df.to_csv(combined_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    save_map(station_gdf.assign(dist_to_active_fault_km=distances_km), regional_faults, map_path)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "station_table": str(INPUT_FILE),
                "fault_geojson": str(fault_file),
                "total_stations": int(len(station_df)),
                "global_fault_trace_count": int(len(faults)),
                "regional_fault_trace_count": int(len(regional_faults)),
                "distance_feature": "dist_to_active_fault_km",
                "projection": LOCAL_CRS,
                "study_buffer_degrees": STUDY_BUFFER_DEGREES,
                "distance_summary_km": station_df["dist_to_active_fault_km"].describe().to_dict(),
                "note": "The GEM global database may not contain every mapped submarine fault in the South China Sea. Treat this as a global active-fault proximity feature.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFinished")
    print(f"Distance range: {distances_km.min():.2f} to {distances_km.max():.2f} km")
    print(f"Within 50 km: {int(station_df['near_active_fault_50km'].sum())}")
    print(f"Within 100 km: {int(station_df['near_active_fault_100km'].sum())}")
    print("Saved:")
    for path in [feature_path, combined_path, map_path, summary_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
