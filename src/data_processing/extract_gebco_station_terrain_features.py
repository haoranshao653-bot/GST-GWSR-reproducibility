r"""
Extract GEBCO bathymetry and terrain features for South China Sea heat-flow stations.

Input:
  1. A GEBCO GeoTIFF placed anywhere under C:\Users\shen\Desktop\xinjiashuju
  2. The original heat_flow_stations.csv

Output:
  C:\Users\shen\Desktop\xinjiashuju\gebco_station_terrain_features

The original station CSV is never overwritten.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform
except ImportError as error:
    raise SystemExit(
        "Missing dependency: rasterio. Install it in this interpreter with:\n"
        "  pip install rasterio\n"
        f"Details: {error}"
    )


DATA_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju")
OUTPUT_DIR = DATA_DIR / "gebco_station_terrain_features"
STATION_FILE = Path(
    r"C:\Users\shen\Desktop\hiyangshuju\qinxuelunwenfangfa\heat_flow\heat_flow_stations.csv"
)
COORD_LON = "lon"
COORD_LAT = "lat"


def find_gebco_geotiff(data_dir: Path) -> Path:
    """Find exactly one GeoTIFF below the user-selected data folder."""
    tif_files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif or .tiff file found under: {data_dir}\n"
            "Put the extracted GEBCO GeoTIFF in this folder, then run again."
        )
    if len(tif_files) > 1:
        listed = "\n".join(f"  - {path}" for path in tif_files)
        raise RuntimeError(
            "More than one GeoTIFF was found. Keep only the GEBCO bathymetry GeoTIFF "
            "in this folder temporarily, or edit GEOTIFF_FILE in the script.\n" + listed
        )
    return tif_files[0]


def finite_values(array: np.ndarray, nodata: float | None) -> np.ndarray:
    values = array.astype(float, copy=True)
    if nodata is not None and np.isfinite(nodata):
        values[np.isclose(values, nodata)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def meters_per_pixel(dataset: rasterio.DatasetReader, latitude: float) -> Tuple[float, float]:
    """Return pixel spacing in metres at one station location."""
    x_resolution = abs(dataset.transform.a)
    y_resolution = abs(dataset.transform.e)
    if dataset.crs and dataset.crs.is_geographic:
        lat_rad = np.deg2rad(latitude)
        dx_m = x_resolution * 111_320.0 * max(np.cos(lat_rad), 1e-6)
        dy_m = y_resolution * 110_540.0
    else:
        dx_m = x_resolution
        dy_m = y_resolution
    return dx_m, dy_m


def read_patch(
    dataset: rasterio.DatasetReader, row: int, col: int, size: int
) -> np.ndarray:
    half = size // 2
    nodata = dataset.nodata
    fill_value = nodata if nodata is not None else np.nan
    patch = dataset.read(
        1,
        window=Window(col - half, row - half, size, size),
        boundless=True,
        fill_value=fill_value,
        out_dtype="float64",
    )
    return finite_values(patch, nodata)


def terrain_features(
    patch_3: np.ndarray, patch_5: np.ndarray, dx_m: float, dy_m: float
) -> Dict[str, float]:
    """Calculate locally interpretable terrain quantities from GEBCO elevation."""
    center = patch_3[1, 1]
    if not np.isfinite(center) or not np.isfinite(patch_3).all():
        return {
            "gebco_elevation_m": np.nan,
            "gebco_water_depth_m": np.nan,
            "gebco_slope_deg": np.nan,
            "gebco_laplacian_m_per_km2": np.nan,
            "gebco_roughness_std_m_5x5": np.nan,
            "gebco_tpi_m_5x5": np.nan,
        }

    dz_dx = (patch_3[1, 2] - patch_3[1, 0]) / (2.0 * dx_m)
    dz_dy = (patch_3[2, 1] - patch_3[0, 1]) / (2.0 * dy_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    d2z_dx2 = (patch_3[1, 2] - 2.0 * center + patch_3[1, 0]) / (dx_m**2)
    d2z_dy2 = (patch_3[2, 1] - 2.0 * center + patch_3[0, 1]) / (dy_m**2)
    laplacian_m_per_km2 = (d2z_dx2 + d2z_dy2) * 1_000_000.0

    patch_5_valid = patch_5[np.isfinite(patch_5)]
    roughness = float(np.std(patch_5_valid)) if len(patch_5_valid) >= 20 else np.nan
    mean_5 = float(np.mean(patch_5_valid)) if len(patch_5_valid) >= 20 else np.nan

    return {
        "gebco_elevation_m": float(center),
        "gebco_water_depth_m": float(max(-center, 0.0)),
        "gebco_slope_deg": float(slope_deg),
        "gebco_laplacian_m_per_km2": float(laplacian_m_per_km2),
        "gebco_roughness_std_m_5x5": roughness,
        "gebco_tpi_m_5x5": float(center - mean_5) if np.isfinite(mean_5) else np.nan,
    }


def station_to_raster_coordinates(
    dataset: rasterio.DatasetReader, lon: float, lat: float
) -> Tuple[float, float]:
    if dataset.crs is None:
        raise ValueError("The GEBCO GeoTIFF has no coordinate reference system.")
    if dataset.crs.to_epsg() == 4326:
        return lon, lat
    x, y = transform("EPSG:4326", dataset.crs, [lon], [lat])
    return x[0], y[0]


def robust_limits(values: pd.Series) -> Tuple[float, float]:
    """Avoid a few extreme stations flattening the station maps."""
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return 0.0, 1.0
    low, high = np.percentile(clean, [2, 98])
    if np.isclose(low, high):
        low, high = float(clean.min()), float(clean.max())
    if np.isclose(low, high):
        high = low + 1.0
    return float(low), float(high)


def save_feature_visualization(frame: pd.DataFrame, output_path: Path) -> None:
    """Create one compact map-and-distribution overview for visual inspection."""
    valid = frame[frame["gebco_valid"]].copy()
    fields = [
        ("gebco_water_depth_m", "GEBCO 水深 (m)", "viridis"),
        ("gebco_slope_deg", "局部坡度 (度)", "magma"),
        ("gebco_roughness_std_m_5x5", "局部粗糙度 (m)", "cividis"),
        ("gebco_laplacian_m_per_km2", "局部曲率代理 (m/km²)", "coolwarm"),
        ("gebco_tpi_m_5x5", "地形位置指数 TPI (m)", "RdBu_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for axis, (field, title, cmap) in zip(axes[:5], fields):
        vmin, vmax = robust_limits(valid[field])
        scatter = axis.scatter(
            valid[COORD_LON], valid[COORD_LAT], c=valid[field], s=18,
            cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="none",
        )
        invalid = frame[~frame["gebco_valid"]]
        if len(invalid):
            axis.scatter(invalid[COORD_LON], invalid[COORD_LAT], marker="x", c="red", s=28, label="缺失")
            axis.legend(loc="best", frameon=False)
        axis.set_title(title)
        axis.set_xlabel("经度")
        axis.set_ylabel("纬度")
        axis.grid(alpha=0.25)
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=8)

    histogram_axis = axes[5]
    histogram_axis.hist(valid["gebco_water_depth_m"].dropna(), bins=35, color="#2878b5", edgecolor="white")
    histogram_axis.set_title("站点 GEBCO 水深分布")
    histogram_axis.set_xlabel("水深 (m)")
    histogram_axis.set_ylabel("站点数")
    histogram_axis.grid(axis="y", alpha=0.25)
    fig.suptitle("南海热流站点的 GEBCO 地形特征", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not STATION_FILE.exists():
        raise FileNotFoundError(f"Original station file not found: {STATION_FILE}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    geotiff_file = find_gebco_geotiff(DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    station_df = pd.read_csv(STATION_FILE)
    required = {COORD_LON, COORD_LAT}
    missing = required - set(station_df.columns)
    if missing:
        raise KeyError(f"Station CSV has no required coordinate columns: {sorted(missing)}")

    print("=" * 72)
    print("GEBCO station terrain-feature extraction")
    print("=" * 72)
    print(f"Station file: {STATION_FILE}")
    print(f"GEBCO GeoTIFF: {geotiff_file}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Station count: {len(station_df)}")

    records = []
    with rasterio.open(geotiff_file) as dataset:
        print(f"Raster CRS: {dataset.crs}")
        print(f"Raster shape: {dataset.width} columns x {dataset.height} rows")
        print(f"Raster bounds: {dataset.bounds}")

        for index, station in station_df.iterrows():
            lon = float(station[COORD_LON])
            lat = float(station[COORD_LAT])
            try:
                x, y = station_to_raster_coordinates(dataset, lon, lat)
                row, col = dataset.index(x, y)
                in_bounds = 0 <= row < dataset.height and 0 <= col < dataset.width
                if not in_bounds:
                    feature = terrain_features(
                        np.full((3, 3), np.nan), np.full((5, 5), np.nan), 1.0, 1.0
                    )
                    feature["gebco_valid"] = False
                else:
                    dx_m, dy_m = meters_per_pixel(dataset, lat)
                    feature = terrain_features(
                        read_patch(dataset, row, col, 3),
                        read_patch(dataset, row, col, 5),
                        dx_m,
                        dy_m,
                    )
                    feature["gebco_valid"] = bool(np.isfinite(feature["gebco_elevation_m"]))
            except Exception as error:
                feature = terrain_features(
                    np.full((3, 3), np.nan), np.full((5, 5), np.nan), 1.0, 1.0
                )
                feature["gebco_valid"] = False
                feature["gebco_error"] = str(error)[:160]
            else:
                feature["gebco_error"] = ""

            records.append(feature)
            if (index + 1) % 100 == 0 or index + 1 == len(station_df):
                print(f"  processed {index + 1}/{len(station_df)}")

    feature_df = pd.DataFrame(records)
    combined_df = pd.concat([station_df.reset_index(drop=True), feature_df], axis=1)
    valid_count = int(combined_df["gebco_valid"].sum())

    feature_path = OUTPUT_DIR / "gebco_station_01_extracted_features.csv"
    combined_path = OUTPUT_DIR / "heat_flow_stations_with_gebco_terrain.csv"
    invalid_path = OUTPUT_DIR / "gebco_station_02_invalid_or_missing_points.csv"
    summary_path = OUTPUT_DIR / "gebco_station_03_extraction_summary.json"
    figure_path = OUTPUT_DIR / "gebco_station_04_feature_maps.png"

    feature_columns = [COORD_LON, COORD_LAT] + list(feature_df.columns)
    combined_df[feature_columns].to_csv(feature_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    combined_df[~combined_df["gebco_valid"]].to_csv(
        invalid_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    numeric_feature_cols = [
        "gebco_elevation_m", "gebco_water_depth_m", "gebco_slope_deg",
        "gebco_laplacian_m_per_km2", "gebco_roughness_std_m_5x5", "gebco_tpi_m_5x5",
    ]
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "station_file": str(STATION_FILE),
                "gebco_geotiff": str(geotiff_file),
                "total_stations": int(len(combined_df)),
                "valid_gebco_stations": valid_count,
                "invalid_gebco_stations": int(len(combined_df) - valid_count),
                "feature_definitions": {
                    "gebco_elevation_m": "GEBCO elevation at the station. Negative values mean seafloor depth.",
                    "gebco_water_depth_m": "max(-elevation, 0), in metres.",
                    "gebco_slope_deg": "Local terrain slope calculated from a 3x3 raster window.",
                    "gebco_laplacian_m_per_km2": "Local curvature proxy from the 3x3 elevation Laplacian.",
                    "gebco_roughness_std_m_5x5": "Elevation standard deviation within a 5x5 raster window.",
                    "gebco_tpi_m_5x5": "Center elevation minus 5x5 local mean elevation.",
                },
                "feature_summary": combined_df[numeric_feature_cols].describe().to_dict(),
                "note": "Do not overwrite the original topo feature blindly. Compare and screen correlations before model experiments.",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    save_feature_visualization(combined_df, figure_path)

    print("\nFinished")
    print(f"Valid GEBCO extraction: {valid_count}/{len(combined_df)}")
    print("Saved:")
    print(f"  {feature_path}")
    print(f"  {combined_path}")
    print(f"  {invalid_path}")
    print(f"  {summary_path}")
    print(f"  {figure_path}")


if __name__ == "__main__":
    main()
