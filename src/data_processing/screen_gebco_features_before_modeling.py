"""Screen GEBCO-derived terrain features before RF and GW-SR experiments.

This script does not train a model. It checks whether the new GEBCO variables
provide information beyond the original topo feature and produces a model-ready
station table with a land/non-marine indicator.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


DATA_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju")
INPUT_FILE = DATA_DIR / "gebco_station_terrain_features" / "heat_flow_stations_with_gebco_terrain.csv"
OUTPUT_DIR = DATA_DIR / "gebco_feature_screening"
TARGET = "heat_flow"
ORIGINAL_FEATURES = ["topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano"]
GEBCO_FEATURES = [
    "gebco_elevation_m",
    "gebco_water_depth_m",
    "gebco_slope_deg",
    "gebco_laplacian_m_per_km2",
    "gebco_roughness_std_m_5x5",
    "gebco_tpi_m_5x5",
]


def save_correlation_heatmap(correlation: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.index)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(correlation.index, fontsize=9)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")

    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]
            if np.isfinite(value) and abs(value) >= 0.55:
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    axis.set_title("原始变量、GEBCO 地形变量与热流的 Pearson 相关性")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def robust_limits(values: pd.Series) -> tuple[float, float]:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(values, [1, 99])
    if np.isclose(lower, upper):
        lower, upper = float(values.min()), float(values.max())
    return float(lower), float(upper)


def save_topo_comparison_plot(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    valid = frame.dropna(subset=["topo", "gebco_elevation_m", TARGET]).copy()
    colors = np.where(valid["gebco_is_land_or_nonmarine"], "#d55e00", "#0072b2")

    axis = axes[0]
    axis.scatter(valid["topo"], valid["gebco_elevation_m"], c=colors, alpha=0.65, s=18, edgecolors="none")
    axis.set_xlabel("原始 topo")
    axis.set_ylabel("GEBCO elevation (m)")
    axis.set_title("原始 topo 与 GEBCO 高程对比\n蓝色：海底；橙色：陆地/非海点")
    axis.grid(alpha=0.25)

    axis = axes[1]
    for is_land, group in valid.groupby("gebco_is_land_or_nonmarine"):
        label = "陆地/非海点" if is_land else "海底点"
        color = "#d55e00" if is_land else "#0072b2"
        axis.hist(group[TARGET], bins=25, alpha=0.6, color=color, label=label)
    axis.set_xlabel("热流 (mW/m²)")
    axis.set_ylabel("站点数")
    axis.set_title("陆地/海底站点的热流分布")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    required = set(ORIGINAL_FEATURES + GEBCO_FEATURES + [TARGET, "lon", "lat", "gebco_valid"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Input data is missing columns: {missing}")

    # Positive GEBCO elevation is land. Near-zero points are included because
    # they are indistinguishable from coastline/very shallow sea at this scale.
    df["gebco_is_land_or_nonmarine"] = df["gebco_elevation_m"] >= 0
    df["gebco_is_seafloor"] = df["gebco_elevation_m"] < 0
    df["gebco_depth_class"] = pd.cut(
        df["gebco_water_depth_m"],
        bins=[-1, 0, 200, 1000, 3000, np.inf],
        labels=["land_or_zero", "shelf_0_200m", "slope_200_1000m", "deep_1000_3000m", "abyss_3000m_plus"],
    )

    numeric_features = ORIGINAL_FEATURES + GEBCO_FEATURES + [TARGET]
    pearson = df[numeric_features].corr(method="pearson")
    spearman = df[numeric_features].corr(method="spearman")
    topo_gebco_pearson = float(pearson.loc["topo", "gebco_elevation_m"])
    topo_gebco_spearman = float(spearman.loc["topo", "gebco_elevation_m"])

    screening_rows = []
    for feature in GEBCO_FEATURES:
        screening_rows.append(
            {
                "feature": feature,
                "missing_count": int(df[feature].isna().sum()),
                "pearson_with_heat_flow": float(pearson.loc[feature, TARGET]),
                "spearman_with_heat_flow": float(spearman.loc[feature, TARGET]),
                "abs_pearson_with_original_topo": float(abs(pearson.loc[feature, "topo"])),
                "abs_spearman_with_original_topo": float(abs(spearman.loc[feature, "topo"])),
                "screening_note": "Potentially redundant with topo" if abs(pearson.loc[feature, "topo"]) >= 0.90 else "Retain for model comparison",
            }
        )
    screening_df = pd.DataFrame(screening_rows)

    enriched_path = OUTPUT_DIR / "gebco_screened_01_stations_with_land_flag.csv"
    pearson_path = OUTPUT_DIR / "gebco_screened_02_pearson_correlation.csv"
    spearman_path = OUTPUT_DIR / "gebco_screened_03_spearman_correlation.csv"
    screening_path = OUTPUT_DIR / "gebco_screened_04_feature_screening.csv"
    heatmap_path = OUTPUT_DIR / "gebco_screened_05_correlation_heatmap.png"
    comparison_path = OUTPUT_DIR / "gebco_screened_06_topo_gebco_quality_check.png"
    summary_path = OUTPUT_DIR / "gebco_screened_07_summary.json"

    df.to_csv(enriched_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    pearson.to_csv(pearson_path, encoding="utf-8-sig", float_format="%.6f")
    spearman.to_csv(spearman_path, encoding="utf-8-sig", float_format="%.6f")
    screening_df.to_csv(screening_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    save_correlation_heatmap(pearson, heatmap_path)
    save_topo_comparison_plot(df, comparison_path)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_file": str(INPUT_FILE),
                "total_stations": int(len(df)),
                "valid_gebco_stations": int(df["gebco_valid"].sum()),
                "land_or_nonmarine_count": int(df["gebco_is_land_or_nonmarine"].sum()),
                "seafloor_count": int(df["gebco_is_seafloor"].sum()),
                "topo_vs_gebco_elevation": {
                    "pearson": topo_gebco_pearson,
                    "spearman": topo_gebco_spearman,
                    "interpretation": "Absolute correlation >= 0.90 indicates the two raw elevation variables are probably redundant.",
                },
                "rule": "Do not include both topo and gebco_elevation_m as raw predictors if they are highly correlated. Test derived terrain features as a group.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("GEBCO feature screening complete")
    print("=" * 72)
    print(f"Stations: {len(df)}")
    print(f"Land/non-marine: {int(df['gebco_is_land_or_nonmarine'].sum())}")
    print(f"Seafloor: {int(df['gebco_is_seafloor'].sum())}")
    print(f"topo vs GEBCO elevation: Pearson={topo_gebco_pearson:.4f}, Spearman={topo_gebco_spearman:.4f}")
    print("\nGEBCO variable screening:")
    print(screening_df.to_string(index=False))
    print("\nSaved:")
    for path in [enriched_path, pearson_path, spearman_path, screening_path, heatmap_path, comparison_path, summary_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
