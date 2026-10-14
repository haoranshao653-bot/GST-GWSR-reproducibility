"""Extract the EMAG2 v3 sea-level magnetic anomaly at each heat-flow station.

Only the raw magnetic anomaly is extracted in this first pass. This avoids
adding several correlated magnetic derivatives before verifying that the
physical magnetic signal improves the prediction problem.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.warp import transform
except ImportError as error:
    raise SystemExit(
        "Missing dependency: rasterio. Install it with: pip install rasterio\n"
        f"Details: {error}"
    )

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


DATA_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju")
EMAG2_FILE = DATA_DIR / "EMAG2_V3_SeaLevel_DataTiff.tif"
INPUT_FILE = DATA_DIR / "gebco_feature_screening" / "gebco_screened_01_stations_with_land_flag.csv"
OUTPUT_DIR = DATA_DIR / "emag2_station_features"
LON_COLUMN = "lon"
LAT_COLUMN = "lat"
TARGET = "heat_flow"
NODATA_VALUE = 99999.0


def transform_coordinates(dataset: rasterio.DatasetReader, lons: np.ndarray, lats: np.ndarray) -> list[tuple[float, float]]:
    """Convert WGS84 station coordinates to the GeoTIFF coordinate system."""
    if dataset.crs is None:
        raise ValueError("EMAG2 GeoTIFF has no coordinate reference system.")
    if dataset.crs.to_epsg() == 4326:
        return list(zip(lons, lats))
    xs, ys = transform("EPSG:4326", dataset.crs, lons.tolist(), lats.tolist())
    return list(zip(xs, ys))


def is_valid_value(value: float, raster_nodata: float | None) -> bool:
    if not np.isfinite(value):
        return False
    if np.isclose(value, NODATA_VALUE):
        return False
    if raster_nodata is not None and np.isfinite(raster_nodata) and np.isclose(value, raster_nodata):
        return False
    return True


def save_visualization(frame: pd.DataFrame, output_path: Path) -> None:
    valid = frame[frame["emag2_valid"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axis = axes[0]
    if len(valid):
        low, high = np.percentile(valid["emag2_sealevel_anomaly_nt"], [2, 98])
        scatter = axis.scatter(
            valid[LON_COLUMN], valid[LAT_COLUMN], c=valid["emag2_sealevel_anomaly_nt"],
            s=20, cmap="RdBu_r", vmin=low, vmax=high, edgecolors="none",
        )
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("磁异常 (nT)")
    invalid = frame[~frame["emag2_valid"]]
    if len(invalid):
        axis.scatter(invalid[LON_COLUMN], invalid[LAT_COLUMN], c="black", marker="x", s=30, label="缺失")
        axis.legend(frameon=False)
    axis.set_title("热流站点 EMAG2 海平面磁异常")
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.grid(alpha=0.25)

    axis = axes[1]
    axis.hist(valid["emag2_sealevel_anomaly_nt"].dropna(), bins=35, color="#3b7ddd", edgecolor="white")
    axis.axvline(0, color="#d55e00", linestyle="--", lw=1.5)
    axis.set_title("EMAG2 磁异常分布")
    axis.set_xlabel("磁异常 (nT)")
    axis.set_ylabel("站点数")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Station table not found: {INPUT_FILE}")
    if not EMAG2_FILE.exists():
        raise FileNotFoundError(f"EMAG2 GeoTIFF not found: {EMAG2_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    missing = {LON_COLUMN, LAT_COLUMN, TARGET} - set(df.columns)
    if missing:
        raise KeyError(f"Input table is missing columns: {sorted(missing)}")

    print("=" * 72)
    print("EMAG2 sea-level magnetic-anomaly extraction")
    print("=" * 72)
    print(f"Station table: {INPUT_FILE}")
    print(f"EMAG2 GeoTIFF: {EMAG2_FILE}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Station count: {len(df)}")

    lons = df[LON_COLUMN].to_numpy(dtype=float)
    lats = df[LAT_COLUMN].to_numpy(dtype=float)
    values = []
    valid_flags = []
    with rasterio.open(EMAG2_FILE) as dataset:
        print(f"Raster CRS: {dataset.crs}")
        print(f"Raster shape: {dataset.width} columns x {dataset.height} rows")
        print(f"Raster bounds: {dataset.bounds}")
        points = transform_coordinates(dataset, lons, lats)
        for index, sample in enumerate(dataset.sample(points, indexes=1), start=1):
            value = float(sample[0])
            valid = is_valid_value(value, dataset.nodata)
            values.append(value if valid else np.nan)
            valid_flags.append(valid)
            if index % 100 == 0 or index == len(df):
                print(f"  processed {index}/{len(df)}")

    df["emag2_sealevel_anomaly_nt"] = values
    df["emag2_valid"] = valid_flags
    valid_count = int(df["emag2_valid"].sum())

    feature_path = OUTPUT_DIR / "emag2_station_01_extracted_feature.csv"
    combined_path = OUTPUT_DIR / "heat_flow_stations_with_gebco_and_emag2.csv"
    invalid_path = OUTPUT_DIR / "emag2_station_02_invalid_or_missing_points.csv"
    figure_path = OUTPUT_DIR / "emag2_station_03_feature_map.png"
    summary_path = OUTPUT_DIR / "emag2_station_04_extraction_summary.json"

    df[[LON_COLUMN, LAT_COLUMN, "emag2_sealevel_anomaly_nt", "emag2_valid"]].to_csv(
        feature_path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    df.to_csv(combined_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    df[~df["emag2_valid"]].to_csv(invalid_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    save_visualization(df, figure_path)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "station_table": str(INPUT_FILE),
                "emag2_geotiff": str(EMAG2_FILE),
                "total_stations": int(len(df)),
                "valid_stations": valid_count,
                "invalid_stations": int(len(df) - valid_count),
                "feature": "emag2_sealevel_anomaly_nt",
                "unit": "nT",
                "source": "EMAG2 v3 Sea Level magnetic anomaly grid",
                "note": "Only the raw magnetic anomaly is extracted. Screen this feature before adding any magnetic derivatives.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFinished")
    print(f"Valid EMAG2 extraction: {valid_count}/{len(df)}")
    print("Saved:")
    for path in [feature_path, combined_path, invalid_path, figure_path, summary_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
