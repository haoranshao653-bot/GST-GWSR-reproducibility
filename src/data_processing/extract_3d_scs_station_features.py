"""Extract station-scale features from the GFZ 3D-SCS lithospheric model.

This script performs feature extraction only.  It does not train a predictive
model.  The output retains flags for the published core model and its extended
boundary area so later comparisons can avoid treating extrapolated boundary
values as equal-quality observations.
"""

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator


STATION_FILE = (
    r"C:\Users\shen\Desktop\xinjiashuju\active_fault_station_features"
    r"\heat_flow_stations_with_fault_distance.csv"
)
MODEL_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju\2024-031_Yan-et-al_data")
OUTPUT_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju\3d_scs_station_features")

# Published bounds from the GFZ data description, WGS84 / UTM Zone 50N, metres.
CORE_BOUNDS = (-120_000.0, 840_000.0, 1_060_000.0, 2_320_000.0)
EXTENDED_BOUNDS = (-1_130_000.0, 1_750_000.0, 0.0, 3_360_000.0)
VS_DEPTHS_KM = [50, 100, 145, 190, 250]

LAYER_FILES = {
    "scs_top_water_elev_m": "2024-031_Yan_01_top_water.xyz",
    "scs_seafloor_elev_m": "2024-031_Yan_02_top_sediment.xyz",
    "scs_top_oceanic_crust_elev_m": "2024-031_Yan_03_top_oceanic_crust.xyz",
    "scs_top_continental_crust_elev_m": "2024-031_Yan_04_top_continent_crust.xyz",
    "scs_moho_elev_m": "2024-031_Yan_05_top_upper_lithos_mantle.xyz",
    "scs_lower_lithosphere_top_elev_m": "2024-031_Yan_06_top_lower_lithos_mantle.xyz",
    "scs_lab_elev_m": "2024-031_Yan_07_top_sublithos_mantle.xyz",
}


def configure_plotting():
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_numeric_columns(path, column_count):
    """Read numeric XYZ-like rows while ignoring optional Petrel-style headers."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) < column_count:
                continue
            try:
                rows.append([float(value) for value in tokens[:column_count]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows were found in: {path}")
    return np.asarray(rows, dtype=float)


def load_utm_grid(path, value_column):
    """Load an x-y-value XYZ file into an interpolator and its native extent."""
    data = load_numeric_columns(path, 3)
    x_values = np.unique(data[:, 0])
    y_values = np.unique(data[:, 1])
    values = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    x_index = np.searchsorted(x_values, data[:, 0])
    y_index = np.searchsorted(y_values, data[:, 1])
    values[y_index, x_index] = data[:, 2]
    interpolator = RegularGridInterpolator(
        (y_values, x_values), values, method="linear", bounds_error=False, fill_value=np.nan
    )
    return interpolator, {
        "feature": value_column,
        "x_min": float(x_values.min()),
        "x_max": float(x_values.max()),
        "y_min": float(y_values.min()),
        "y_max": float(y_values.max()),
        "grid_spacing_m": float(np.median(np.diff(x_values))),
    }


def load_gravity_grid(path):
    """Load the 6-km-altitude gravity disturbance XYZ file."""
    raw_data = load_numeric_columns(path, 4)
    data = raw_data[:, [0, 1, 3]]
    x_values = np.unique(data[:, 0])
    y_values = np.unique(data[:, 1])
    values = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    values[np.searchsorted(y_values, data[:, 1]), np.searchsorted(x_values, data[:, 0])] = data[:, 2]
    return RegularGridInterpolator(
        (y_values, x_values), values, method="linear", bounds_error=False, fill_value=np.nan
    )


def load_vs_interpolators(path):
    """Create one longitude-latitude interpolator for every requested depth."""
    data = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "depth_km", "vs_km_s"])
    interpolators = {}
    for depth in VS_DEPTHS_KM:
        subset = data.loc[data["depth_km"] == depth]
        lon_values = np.sort(subset["lon"].unique())
        lat_values = np.sort(subset["lat"].unique())
        values = np.full((len(lat_values), len(lon_values)), np.nan, dtype=float)
        values[
            np.searchsorted(lat_values, subset["lat"].to_numpy()),
            np.searchsorted(lon_values, subset["lon"].to_numpy()),
        ] = subset["vs_km_s"].to_numpy()
        interpolators[depth] = RegularGridInterpolator(
            (lat_values, lon_values), values, method="linear", bounds_error=False, fill_value=np.nan
        )
    return interpolators


def nonnegative_difference(upper, lower):
    result = upper - lower
    return result.where(result >= 0)


def save_maps(df, path):
    plot_specs = [
        ("scs_sediment_thickness_m", "3D-SCS sediment thickness (m)", "viridis"),
        ("scs_moho_depth_m", "3D-SCS Moho depth (m)", "viridis"),
        ("scs_lab_depth_m", "3D-SCS LAB depth (m)", "viridis"),
        ("scs_lower_lithosphere_density_kg_m3", "Lower lithosphere density (kg/m3)", "plasma"),
        ("scs_vs_100km_km_s", "Shear-wave velocity at 100 km (km/s)", "magma"),
        ("scs_in_core_model", "Published core-model coverage", "Greens"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for axis, (column, title, cmap) in zip(axes.ravel(), plot_specs):
        values = df[column]
        image = axis.scatter(df["lon"], df["lat"], c=values, s=20, cmap=cmap, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("Station features extracted from the GFZ 3D-SCS model", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(STATION_FILE):
        raise FileNotFoundError(f"Station file not found: {STATION_FILE}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"3D-SCS folder not found: {MODEL_DIR}")

    missing_files = [name for name in LAYER_FILES.values() if not (MODEL_DIR / name).exists()]
    required_other = [
        "2024-031_Yan_D_lower_lithospheric_mantle.xyz",
        "2024-031_Yan_G_EIGEN-6C4_10km_disturbance_6km.xyz",
        "2024-031_Yan_Vs_Tang_Zheng.xyz",
    ]
    missing_files.extend(name for name in required_other if not (MODEL_DIR / name).exists())
    if missing_files:
        raise FileNotFoundError("Missing 3D-SCS files: " + ", ".join(missing_files))

    print("=" * 72)
    print("GFZ 3D-SCS station feature extraction")
    print("=" * 72)
    print(f"Station file: {STATION_FILE}")
    print(f"3D-SCS folder: {MODEL_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    stations = pd.read_csv(STATION_FILE)
    required_columns = {"lon", "lat", "heat_flow"}
    absent = required_columns - set(stations.columns)
    if absent:
        raise ValueError(f"Station table is missing columns: {sorted(absent)}")
    print(f"Stations: {len(stations)}")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    easting, northing = transformer.transform(stations["lon"].to_numpy(), stations["lat"].to_numpy())
    stations["scs_utm50_easting_m"] = easting
    stations["scs_utm50_northing_m"] = northing

    x0, x1, y0, y1 = CORE_BOUNDS
    stations["scs_in_core_model"] = (
        (easting >= x0) & (easting <= x1) & (northing >= y0) & (northing <= y1)
    )
    x0, x1, y0, y1 = EXTENDED_BOUNDS
    stations["scs_in_extended_model"] = (
        (easting >= x0) & (easting <= x1) & (northing >= y0) & (northing <= y1)
    )

    utm_points = np.column_stack([northing, easting])
    metadata = {}
    for column, filename in LAYER_FILES.items():
        print(f"  extracting {column}")
        interpolator, grid_info = load_utm_grid(MODEL_DIR / filename, column)
        stations[column] = interpolator(utm_points)
        metadata[column] = grid_info

    print("  extracting lower-lithosphere density")
    density_interpolator, density_info = load_utm_grid(
        MODEL_DIR / "2024-031_Yan_D_lower_lithospheric_mantle.xyz",
        "scs_lower_lithosphere_density_kg_m3",
    )
    stations["scs_lower_lithosphere_density_kg_m3"] = density_interpolator(utm_points)
    metadata["scs_lower_lithosphere_density_kg_m3"] = density_info

    print("  extracting gravity disturbance")
    gravity_interpolator = load_gravity_grid(
        MODEL_DIR / "2024-031_Yan_G_EIGEN-6C4_10km_disturbance_6km.xyz"
    )
    stations["scs_free_air_gravity_disturbance_mgal"] = gravity_interpolator(utm_points)

    print("  extracting shear-wave velocity profiles")
    vs_interpolators = load_vs_interpolators(MODEL_DIR / "2024-031_Yan_Vs_Tang_Zheng.xyz")
    geographic_points = np.column_stack([stations["lat"], stations["lon"]])
    for depth, interpolator in vs_interpolators.items():
        stations[f"scs_vs_{depth}km_km_s"] = interpolator(geographic_points)
    stations["scs_vs_250_minus_100_km_s"] = (
        stations["scs_vs_250km_km_s"] - stations["scs_vs_100km_km_s"]
    )

    # The shallower of the oceanic and continental interfaces is the local crystalline basement.
    stations["scs_crystalline_basement_elev_m"] = stations[
        ["scs_top_oceanic_crust_elev_m", "scs_top_continental_crust_elev_m"]
    ].max(axis=1)
    stations["scs_water_depth_m"] = (-stations["scs_seafloor_elev_m"]).clip(lower=0)
    stations["scs_sediment_thickness_m"] = nonnegative_difference(
        stations["scs_seafloor_elev_m"], stations["scs_crystalline_basement_elev_m"]
    )
    stations["scs_moho_depth_m"] = -stations["scs_moho_elev_m"]
    stations["scs_lab_depth_m"] = -stations["scs_lab_elev_m"]
    stations["scs_crystalline_crust_thickness_m"] = nonnegative_difference(
        stations["scs_crystalline_basement_elev_m"], stations["scs_moho_elev_m"]
    )
    stations["scs_lithospheric_mantle_thickness_m"] = nonnegative_difference(
        stations["scs_moho_elev_m"], stations["scs_lab_elev_m"]
    )

    feature_columns = [column for column in stations.columns if column.startswith("scs_")]
    valid_core = stations["scs_in_core_model"]
    summary = {
        "source": "GFZ 3D-SCS, DOI: 10.5880/fidgeo.2024.031",
        "station_count": int(len(stations)),
        "within_published_core_model": int(valid_core.sum()),
        "within_extended_model": int(stations["scs_in_extended_model"].sum()),
        "core_bounds_utm50n_m": CORE_BOUNDS,
        "extended_bounds_utm50n_m": EXTENDED_BOUNDS,
        "extracted_feature_columns": feature_columns,
        "structural_grid_metadata": metadata,
        "interpretation_note": (
            "Use scs_in_core_model to restrict any main-model test. Values in the extended boundary "
            "area are retained for mapping only unless separately validated."
        ),
    }

    output_features = OUTPUT_DIR / "3d_scs_01_extracted_features.csv"
    output_stations = OUTPUT_DIR / "heat_flow_stations_with_3d_scs_features.csv"
    output_definition = OUTPUT_DIR / "3d_scs_02_feature_definitions.csv"
    output_summary = OUTPUT_DIR / "3d_scs_03_extraction_summary.json"
    output_map = OUTPUT_DIR / "3d_scs_04_feature_maps.png"

    stations[feature_columns].to_csv(output_features, index=False, float_format="%.5f")
    stations.to_csv(output_stations, index=False, float_format="%.5f")

    definitions = pd.DataFrame(
        [
            ("scs_sediment_thickness_m", "Seafloor to crystalline-basement thickness"),
            ("scs_moho_depth_m", "Depth of the top upper-lithospheric-mantle interface"),
            ("scs_lab_depth_m", "Depth of the top sub-lithospheric-mantle interface (LAB proxy)"),
            ("scs_crystalline_crust_thickness_m", "Crystalline basement to Moho thickness"),
            ("scs_lithospheric_mantle_thickness_m", "Moho to LAB thickness"),
            ("scs_lower_lithosphere_density_kg_m3", "3D gravity-inversion lower-lithosphere density"),
            ("scs_free_air_gravity_disturbance_mgal", "EIGEN-6C4 gravity disturbance at 6 km altitude"),
            ("scs_vs_*km_km_s", "Tang and Zheng shear-wave velocity at fixed depth"),
            ("scs_in_core_model", "Inside published 960 x 1260 km core model"),
            ("scs_in_extended_model", "Inside extended model boundary region"),
        ],
        columns=["feature", "definition"],
    )
    definitions.to_csv(output_definition, index=False)
    with open(output_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    save_maps(stations, output_map)

    print("\nFinished")
    print(f"Published core-model coverage: {int(valid_core.sum())}/{len(stations)}")
    print(f"Extended-model coverage: {int(stations['scs_in_extended_model'].sum())}/{len(stations)}")
    print("Saved:")
    for path in [output_features, output_stations, output_definition, output_summary, output_map]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
