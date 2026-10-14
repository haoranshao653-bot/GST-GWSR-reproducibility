"""Create two publication-ready figures for the South China Sea study.

Figure 1: independent study-area and heat-flow-station map.
Figure 2: independent 2 x 4 collection of the eight geophysical predictors.

The script reads the original physical-unit GeoTIFF files. It does not plot
standardized model inputs and it does not interpolate the station table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter


# -----------------------------------------------------------------------------
# 1. Paths and plotting extent
# -----------------------------------------------------------------------------

DATA_ROOT = Path(r"C:\Users\shen\Desktop\hiyangshuju\qinxuelunwenfangfa")
ORIGINAL_ROOT = Path(r"C:\Users\shen\Desktop\hiyangshuju\original data\original data")

STATION_CSV = DATA_ROOT / "heat_flow" / "heat_flow_stations.csv"
COASTLINE_SHP = ORIGINAL_ROOT / "ne_10m_coastline" / "ne_10m_coastline.shp"
GEBCO_BACKGROUND = ORIGINAL_ROOT / "gebco_2026_n27.0_s-6.0_w102.0_e125.0_geotiff.tif"

OUTPUT_DIR = Path(r"C:\Users\shen\Documents\海洋热流研究\south_china_sea_maps")

# longitude_min, longitude_max, latitude_min, latitude_max
STUDY_EXTENT = (105.0, 122.0, 0.0, 25.0)

# The display range uses the 2nd-98th percentiles to prevent a few grid-edge
# extremes from flattening spatial contrast. Set False to use the full range.
ROBUST_COLOR_LIMITS = True

# Figure 2 uses the analytical 0.25-degree grids. A light display-only Gaussian
# filter suppresses isolated cell-scale speckling without modifying any source
# file or model input. Set this to 0 to show the raw cells.
FEATURE_DISPLAY_SMOOTHING_SIGMA = 0.75

# Interpolation affects only the rendered image, not the input raster values.
DISPLAY_INTERPOLATION = "bicubic"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    title: str
    unit: str
    relative_path: str
    cmap: str
    diverging_center: Optional[float] = None


FEATURES = (
    FeatureSpec(
        "topo",
        "Seafloor topography",
        "m",
        r"topo\topo_grid_025deg.tif",
        "scs_topography",
        0.0,
    ),
    FeatureSpec(
        "bouguer",
        "Bouguer gravity anomaly",
        "mGal",
        r"bouguer\bouguer_grid_025deg.tif",
        "Spectral_r",
    ),
    FeatureSpec(
        "sediment",
        "Sediment thickness",
        "km",
        r"sediment\sediment_grid_025deg.tif",
        "YlOrBr",
    ),
    FeatureSpec(
        "moho",
        "Moho depth",
        "km",
        r"moho\moho_grid_025deg.tif",
        "viridis_r",
    ),
    FeatureSpec(
        "lab",
        "LAB depth",
        "km",
        r"lab\lab_grid_025deg.tif",
        "magma_r",
    ),
    FeatureSpec(
        "cpd",
        "Curie-point depth",
        "km",
        r"cpd\cpd_grid_025deg.tif",
        "cividis_r",
    ),
    FeatureSpec(
        "lithos_km",
        "Lithospheric thickness",
        "km",
        r"lithos_km\lithos_km_grid_025deg.tif",
        "inferno_r",
    ),
    FeatureSpec(
        "volcano",
        "Distance to Holocene volcanoes",
        "km",
        r"volcano\volcano_grid_025deg.tif",
        "YlGnBu",
    ),
)


# A restrained bathymetry-land palette. With TwoSlopeNorm, sea level is placed
# at the central pale color, negative values are blue, and land is tan-brown.
SCS_TOPOGRAPHY = LinearSegmentedColormap.from_list(
    "scs_topography",
    (
        "#244b6b",
        "#4f7894",
        "#82a8ba",
        "#b9d1d7",
        "#eef1eb",
        "#ddd8cb",
        "#c6b69f",
        "#9b8067",
    ),
    N=256,
)
mpl.colormaps.register(SCS_TOPOGRAPHY, force=True)


# -----------------------------------------------------------------------------
# 2. General helpers
# -----------------------------------------------------------------------------

GEOD = Geod(ellps="WGS84")


def validate_inputs() -> None:
    required = [STATION_CSV, COASTLINE_SHP, GEBCO_BACKGROUND]
    required.extend(DATA_ROOT / spec.relative_path for spec in FEATURES)
    missing = [path for path in required if not path.exists()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required input files are missing:\n{details}")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def longitude_label(value: float, _position=None) -> str:
    suffix = "E" if value >= 0 else "W"
    return f"{abs(value):g}°{suffix}"


def latitude_label(value: float, _position=None) -> str:
    if abs(value) < 1e-9:
        return "0°"
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):g}°{suffix}"


def style_geo_axes(ax, show_xlabels: bool = True, show_ylabels: bool = True) -> None:
    lon_min, lon_max, lat_min, lat_max = STUDY_EXTENT
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#eaf3f5")
    ax.set_xticks([105, 110, 115, 120])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.xaxis.set_major_formatter(FuncFormatter(longitude_label))
    ax.yaxis.set_major_formatter(FuncFormatter(latitude_label))
    ax.tick_params(
        labelbottom=show_xlabels,
        labelleft=show_ylabels,
        labelsize=8.5,
        pad=2,
    )
    ax.grid(color="#ffffff", linewidth=0.55, alpha=0.55, linestyle="--", zorder=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#262626")


def read_cropped_raster(path: Path):
    lon_min, lon_max, lat_min, lat_max = STUDY_EXTENT
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        if src.crs.to_epsg() != 4326:
            raise ValueError(
                f"Expected EPSG:4326 but found {src.crs} in {path}. "
                "Reproject before plotting."
            )
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        window = window.round_offsets().round_lengths()
        array = src.read(1, window=window, masked=True).astype("float64")
        transform = src.window_transform(window)

    data = array.filled(np.nan)
    data[~np.isfinite(data)] = np.nan
    extent = plotting_extent(array, transform)
    return data, extent


def read_high_resolution_background(path: Path, target_width: int = 2200):
    """Read a high-resolution GEBCO window at a publication-efficient size."""
    lon_min, lon_max, lat_min, lat_max = STUDY_EXTENT
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f"Expected an EPSG:4326 raster: {path}")
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        scale = target_width / float(window.width)
        target_height = max(1, int(round(window.height * scale)))
        array = src.read(
            1,
            window=window,
            out_shape=(target_height, target_width),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype("float64")

    data = array.filled(np.nan)
    data[~np.isfinite(data)] = np.nan
    return data, (lon_min, lon_max, lat_min, lat_max)


def smooth_raster_for_display(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply nodata-aware Gaussian smoothing to a display copy of a raster."""
    if sigma <= 0:
        return data
    valid = np.isfinite(data)
    numerator = gaussian_filter(np.where(valid, data, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    smoothed = np.full_like(data, np.nan, dtype=float)
    np.divide(numerator, denominator, out=smoothed, where=denominator > 1e-8)
    return smoothed


def display_limits(data: np.ndarray) -> tuple[float, float]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError("Raster contains no finite values in STUDY_EXTENT.")
    if ROBUST_COLOR_LIMITS:
        low, high = np.nanpercentile(valid, [2.0, 98.0])
    else:
        low, high = np.nanmin(valid), np.nanmax(valid)
    if np.isclose(low, high):
        high = low + 1.0
    return float(low), float(high)


def color_norm(data: np.ndarray, center: Optional[float] = None):
    low, high = display_limits(data)
    if center is not None and low < center < high:
        return TwoSlopeNorm(vmin=low, vcenter=center, vmax=high)
    return Normalize(vmin=low, vmax=high)


def plot_coastline(ax, coastline: gpd.GeoDataFrame, linewidth: float = 0.6) -> None:
    coastline.plot(ax=ax, color="#262626", linewidth=linewidth, zorder=8)


def add_north_arrow(ax, x: float = 0.92, y: float = 0.94, size: float = 0.075) -> None:
    """Draw a compact north arrow in axes coordinates."""
    transform = ax.transAxes
    outer = np.array(
        [
            [x, y],
            [x - size * 0.32, y - size],
            [x, y - size * 0.72],
            [x + size * 0.32, y - size],
        ]
    )
    inner = np.array(
        [
            [x, y - size * 0.08],
            [x, y - size * 0.72],
            [x + size * 0.22, y - size * 0.86],
        ]
    )
    ax.add_patch(
        Polygon(
            outer,
            closed=True,
            transform=transform,
            facecolor="#111111",
            edgecolor="#111111",
            linewidth=0.6,
            zorder=20,
            clip_on=False,
        )
    )
    ax.add_patch(
        Polygon(
            inner,
            closed=True,
            transform=transform,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.4,
            zorder=21,
            clip_on=False,
        )
    )
    ax.text(
        x,
        y + size * 0.12,
        "N",
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        zorder=22,
    )


def add_scale_bar(
    ax,
    length_km: float = 500.0,
    segments: int = 4,
    x_fraction: float = 0.06,
    y_fraction: float = 0.045,
) -> None:
    """Draw a geodesically sized alternating black-white scale bar."""
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    lon0 = lon_min + x_fraction * (lon_max - lon_min)
    lat0 = lat_min + y_fraction * (lat_max - lat_min)
    bar_height = 0.16

    endpoints = []
    for index in range(segments + 1):
        distance_m = length_km * 1000.0 * index / segments
        lon, _lat, _azimuth = GEOD.fwd(lon0, lat0, 90.0, distance_m)
        endpoints.append(lon)

    for index in range(segments):
        left, right = endpoints[index], endpoints[index + 1]
        ax.add_patch(
            Rectangle(
                (left, lat0),
                right - left,
                bar_height,
                facecolor="black" if index % 2 == 0 else "white",
                edgecolor="black",
                linewidth=0.55,
                zorder=30,
            )
        )

    ax.text(endpoints[0], lat0 + bar_height + 0.08, "0", ha="center", va="bottom", fontsize=6.8, zorder=31)
    ax.text(
        endpoints[segments // 2],
        lat0 + bar_height + 0.08,
        f"{length_km / 2:g}",
        ha="center",
        va="bottom",
        fontsize=6.8,
        zorder=31,
    )
    ax.text(
        endpoints[-1],
        lat0 + bar_height + 0.08,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=6.8,
        zorder=31,
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.018,
        0.982,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        zorder=40,
    )


def draw_vertical_colorbar(
    cax,
    cmap,
    norm,
    title: str,
    tick_count: int = 4,
    tick_size: float = 7.0,
    title_size: float = 7.5,
) -> None:
    """Draw a PDF-safe colorbar from an image plus vector labels.

    Matplotlib's QuadMesh colorbar can be misplaced by some PDF renderers when
    it is used inside inset axes. A one-column image is visually identical and
    remains stable in Poppler, Word and LibreOffice.
    """
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    gradient = np.linspace(vmin, vmax, 512, dtype=float).reshape(-1, 1)
    cax.imshow(
        gradient,
        extent=(0.0, 1.0, vmin, vmax),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=1,
    )
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(vmin, vmax)
    cax.set_xticks([])
    locator = MaxNLocator(nbins=tick_count)
    ticks = locator.tick_values(vmin, vmax)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    cax.set_yticks(ticks)
    cax.yaxis.set_ticks_position("left")
    cax.tick_params(axis="y", labelsize=tick_size, length=2.0, pad=1.4)
    cax.set_title(title, fontsize=title_size, pad=5)
    for spine in cax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#333333")

    top_color = cmap(norm(vmax))
    bottom_color = cmap(norm(vmin))
    cax.add_patch(
        Polygon(
            [[0.0, 1.0], [1.0, 1.0], [0.5, 1.075]],
            closed=True,
            transform=cax.transAxes,
            facecolor=top_color,
            edgecolor="#333333",
            linewidth=0.55,
            clip_on=False,
            zorder=3,
        )
    )
    cax.add_patch(
        Polygon(
            [[0.0, 0.0], [1.0, 0.0], [0.5, -0.075]],
            closed=True,
            transform=cax.transAxes,
            facecolor=bottom_color,
            edgecolor="#333333",
            linewidth=0.55,
            clip_on=False,
            zorder=3,
        )
    )


# -----------------------------------------------------------------------------
# 3. Figure 1: independent study-area and station map
# -----------------------------------------------------------------------------

def create_study_area_map(
    stations: pd.DataFrame,
    coastline: gpd.GeoDataFrame,
) -> None:
    # Use the original 15-arc-second GEBCO raster here, rather than the
    # 0.25-degree modelling grid. This removes cell-scale artefacts from the
    # contextual basemap while leaving the analytical predictor unchanged.
    topo, topo_extent = read_high_resolution_background(GEBCO_BACKGROUND)
    topo_norm = color_norm(topo, center=0.0)

    fig, ax = plt.subplots(figsize=(8.8, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.82, bottom=0.08, top=0.975)

    ax.imshow(
        topo,
        extent=topo_extent,
        origin="upper",
        cmap="scs_topography",
        norm=topo_norm,
        interpolation=DISPLAY_INTERPOLATION,
        zorder=1,
    )
    plot_coastline(ax, coastline, linewidth=0.75)

    heat_min, heat_max = np.nanpercentile(stations["heat_flow"], [1.0, 99.0])
    points = ax.scatter(
        stations["lon"],
        stations["lat"],
        c=stations["heat_flow"],
        cmap="YlOrRd",
        norm=Normalize(vmin=float(heat_min), vmax=float(heat_max)),
        s=15,
        edgecolors="white",
        linewidths=0.28,
        alpha=0.88,
        zorder=12,
        # Keep the 1,024 station symbols as vectors. Some PDF backends clip a
        # rasterized scatter layer when bbox_inches="tight" is used.
        rasterized=False,
    )

    style_geo_axes(ax, show_xlabels=True, show_ylabels=True)
    add_north_arrow(ax, x=0.925, y=0.958, size=0.07)
    add_scale_bar(ax, length_km=500, segments=4, x_fraction=0.055, y_fraction=0.042)

    ax.text(
        113.5,
        10.4,
        "South China Sea",
        ha="center",
        va="center",
        fontsize=14,
        fontstyle="italic",
        color="#163b55",
        alpha=0.72,
        zorder=10,
    )
    ax.text(
        0.025,
        0.025,
        f"Heat-flow stations  n = {len(stations):,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.86, "pad": 3.0},
        zorder=35,
    )

    # Keep the heat-flow legend outside the geographic frame so it does not
    # obscure stations or coastline detail.
    cax = fig.add_axes([0.855, 0.16, 0.028, 0.285])
    draw_vertical_colorbar(
        cax,
        cmap=points.cmap,
        norm=points.norm,
        title="Heat flow\n(mW m$^{-2}$)",
        tick_count=5,
        tick_size=8.0,
        title_size=8.5,
    )

    # Regional inset showing where the study window lies in East/Southeast Asia.
    inset = inset_axes(ax, width="33%", height="25%", loc="upper left", borderpad=1.25)
    regional_coast = gpd.read_file(COASTLINE_SHP).cx[90:135, -10:35]
    regional_coast.plot(ax=inset, color="#4c4c4c", linewidth=0.38, zorder=2)
    inset.add_patch(
        Rectangle(
            (STUDY_EXTENT[0], STUDY_EXTENT[2]),
            STUDY_EXTENT[1] - STUDY_EXTENT[0],
            STUDY_EXTENT[3] - STUDY_EXTENT[2],
            facecolor="none",
            edgecolor="#d73027",
            linewidth=1.25,
            zorder=4,
        )
    )
    inset.set_xlim(90, 135)
    inset.set_ylim(-10, 35)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("#edf6f8")
    inset.set_title("Regional location", fontsize=8.5, pad=2)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)

    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"Figure_1_South_China_Sea_heat_flow_stations.{suffix}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Figure 2: independent 2 x 4 geophysical-feature collection
# -----------------------------------------------------------------------------

def create_feature_collection(coastline: gpd.GeoDataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.07, top=0.97, wspace=0.10, hspace=0.13)

    panel_letters = "abcdefgh"
    for index, (ax, spec) in enumerate(zip(axes.flat, FEATURES)):
        data, extent = read_cropped_raster(DATA_ROOT / spec.relative_path)
        norm = color_norm(data, center=spec.diverging_center)
        display_data = smooth_raster_for_display(
            data,
            sigma=FEATURE_DISPLAY_SMOOTHING_SIGMA,
        )

        image = ax.imshow(
            display_data,
            extent=extent,
            origin="upper",
            cmap=spec.cmap,
            norm=norm,
            interpolation=DISPLAY_INTERPOLATION,
            zorder=1,
        )
        plot_coastline(ax, coastline, linewidth=0.48)

        row, column = divmod(index, 4)
        style_geo_axes(
            ax,
            show_xlabels=(row == 1),
            show_ylabels=(column == 0),
        )
        ax.set_title(spec.title, fontsize=10.5, fontweight="bold", pad=5)
        add_panel_label(ax, f"({panel_letters[index]})")
        add_north_arrow(ax, x=0.915, y=0.955, size=0.068)
        add_scale_bar(ax, length_km=500, segments=4, x_fraction=0.05, y_fraction=0.042)

        # A compact vertical colorbar follows the reference figure and remains
        # separate from the scale bar at the lower left.
        cax = inset_axes(
            ax,
            width="4.2%",
            height="25%",
            loc="lower right",
            borderpad=1.05,
        )
        draw_vertical_colorbar(
            cax,
            cmap=image.get_cmap(),
            norm=image.norm,
            title=spec.unit,
            tick_count=3,
            tick_size=6.5,
            title_size=7.2,
        )

    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"Figure_2_Eight_geophysical_features.{suffix}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    validate_inputs()
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(STATION_CSV)
    required_columns = {"lon", "lat", "heat_flow"}
    missing_columns = required_columns.difference(stations.columns)
    if missing_columns:
        raise ValueError(f"Station table is missing columns: {sorted(missing_columns)}")
    stations = stations.loc[
        stations["lon"].between(STUDY_EXTENT[0], STUDY_EXTENT[1])
        & stations["lat"].between(STUDY_EXTENT[2], STUDY_EXTENT[3])
    ].copy()

    coastline = gpd.read_file(COASTLINE_SHP)
    if coastline.crs is None:
        coastline = coastline.set_crs(4326)
    elif coastline.crs.to_epsg() != 4326:
        coastline = coastline.to_crs(4326)
    coastline = coastline.cx[
        STUDY_EXTENT[0] : STUDY_EXTENT[1],
        STUDY_EXTENT[2] : STUDY_EXTENT[3],
    ]

    create_study_area_map(stations, coastline)
    create_feature_collection(coastline)

    print("Created:")
    for path in sorted(OUTPUT_DIR.glob("Figure_*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
