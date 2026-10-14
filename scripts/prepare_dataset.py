"""Build the model-ready station table from public source tables.

The final paper input uses the already harmonized 0.25-degree predictor grids.
This script reproduces the deterministic station filtering, duplicate-coordinate
averaging, nearest-grid extraction, rounding, and audit manifest. It does not
fit a model or inspect any held-out responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import BallTree

FEATURES = ("topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano")
OUTPUT_COLUMNS = ("lon", "lat", "heat_flow", *FEATURES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolate_grid(stations: pd.DataFrame, grid_path: Path, feature: str) -> pd.Series:
    grid = pd.read_csv(grid_path, usecols=["lon", "lat", feature])
    grid = grid.dropna(subset=["lon", "lat", feature]).reset_index(drop=True)
    lats = np.sort(grid["lat"].unique())
    lons = np.sort(grid["lon"].unique())
    values = grid.pivot(index="lat", columns="lon", values=feature).reindex(
        index=lats, columns=lons
    ).to_numpy(dtype=float)
    interpolator = RegularGridInterpolator((lats, lons), values, bounds_error=False, fill_value=None)
    return interpolator(stations[["lat", "lon"]].to_numpy(dtype=float))


def volcano_distance(stations: pd.DataFrame, volcano_path: Path) -> np.ndarray:
    volcanoes = pd.read_csv(volcano_path, encoding="latin1", header=1)
    coordinates = volcanoes[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    tree = BallTree(np.radians(coordinates.to_numpy(dtype=float)), metric="haversine")
    distances, _ = tree.query(np.radians(stations[["lat", "lon"]].to_numpy(dtype=float)), k=1)
    return np.round(distances[:, 0] * 6371.0, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-heat-flow", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_heat_flow)
    required = {"q", "lat", "lon", "environment"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw heat-flow table is missing columns: {sorted(missing)}")

    clipped = raw.loc[
        raw["lon"].between(105, 122) & raw["lat"].between(0, 25)
    ].copy()
    marine = clipped.loc[clipped["environment"].astype(str).str.contains("offshore", case=False, na=False)].copy()
    marine["heat_flow"] = pd.to_numeric(marine["q"], errors="coerce")
    quality = marine.loc[marine["heat_flow"].between(20, 300)].copy()
    retained = quality.loc[quality["heat_flow"] >= 30].copy()
    stations = retained.groupby(["lon", "lat"], as_index=False)["heat_flow"].mean()
    # Duplicate coordinates are averaged at their source precision. The
    # published station table then records coordinates to four decimals.
    stations["lon"] = stations["lon"].round(4)
    stations["lat"] = stations["lat"].round(4)
    stations["heat_flow"] = stations["heat_flow"].round(4)

    for feature in FEATURES:
        grid_path = args.grid_root / feature / f"{feature}_grid_025deg.csv"
        if not grid_path.exists():
            grid_path = args.grid_root / f"{feature}_grid_025deg.csv"
        if not grid_path.exists():
            raise FileNotFoundError(grid_path)
        if feature == "volcano":
            source_path = args.grid_root.parent / "source_products" / "GVP_Volcano_List_Holocene_202605272245.csv"
            stations[feature] = volcano_distance(stations, source_path)
        else:
            stations[feature] = interpolate_grid(stations, grid_path, feature)
    stations = stations.loc[:, OUTPUT_COLUMNS].round(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(args.output, index=False, float_format="%.4f")

    audit = {
        "raw_heat_flow": str(args.raw_heat_flow),
        "raw_heat_flow_sha256": sha256(args.raw_heat_flow),
        "grid_root": str(args.grid_root),
        "grid_files": {
            feature: sha256(
                (args.grid_root / feature / f"{feature}_grid_025deg.csv")
                if (args.grid_root / feature / f"{feature}_grid_025deg.csv").exists()
                else (args.grid_root / f"{feature}_grid_025deg.csv")
            )
            for feature in FEATURES
        },
        "counts": {
            "raw": int(len(raw)),
            "study_area": int(len(clipped)),
            "marine": int(len(marine)),
            "heat_flow_20_300": int(len(quality)),
            "heat_flow_ge_30": int(len(retained)),
            "deduplicated_stations": int(len(stations)),
        },
        "filters": {
            "longitude": [105, 122],
            "latitude": [0, 25],
            "marine_marker": "offshore",
            "heat_flow_range_before_30_cut": [20, 300],
            "minimum_heat_flow": 30,
            "coordinate_rounding_decimals": 4,
            "grid_sampling": "bilinear interpolation on the 0.25-degree grids; volcano uses nearest GVP point and a 6371-km spherical Earth",
        },
        "output_sha256": sha256(args.output),
    }
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
