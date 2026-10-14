"""Screen GFZ 3D-SCS station features before any predictive-model experiment.

The published core model (rather than the extended boundary grid) is the
primary evidence set.  Full-coverage results are saved only as exploratory
context and must not be used alone to justify a main GW-SR feature.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


INPUT_FILE = (
    r"C:\Users\shen\Desktop\xinjiashuju\3d_scs_station_features"
    r"\heat_flow_stations_with_3d_scs_features.csv"
)
OUTPUT_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju\3d_scs_feature_screening")
TARGET = "heat_flow"
ORIGINAL_FEATURES = ["topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano"]
SCS_CANDIDATES = [
    "scs_sediment_thickness_m",
    "scs_moho_depth_m",
    "scs_lab_depth_m",
    "scs_crystalline_crust_thickness_m",
    "scs_lithospheric_mantle_thickness_m",
    "scs_lower_lithosphere_density_kg_m3",
    "scs_free_air_gravity_disturbance_mgal",
    "scs_vs_50km_km_s",
    "scs_vs_100km_km_s",
    "scs_vs_145km_km_s",
    "scs_vs_190km_km_s",
    "scs_vs_250km_km_s",
    "scs_vs_250_minus_100_km_s",
]


def configure_plotting():
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def pairwise_spearman(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan
    return float(spearmanr(valid["x"], valid["y"]).statistic)


def build_screening_table(data, label):
    rows = []
    for feature in SCS_CANDIDATES:
        valid = data[[feature, TARGET]].dropna()
        pearson_heat = valid[feature].corr(valid[TARGET], method="pearson") if len(valid) >= 3 else np.nan
        spearman_heat = pairwise_spearman(valid[feature], valid[TARGET])
        existing_relations = {
            original: data[[feature, original]].dropna()[feature].corr(
                data[[feature, original]].dropna()[original], method="pearson"
            )
            for original in ORIGINAL_FEATURES
        }
        strongest_name = max(existing_relations, key=lambda name: abs(existing_relations[name]))
        strongest_value = existing_relations[strongest_name]
        rows.append(
            {
                "scope": label,
                "feature": feature,
                "valid_count": int(len(valid)),
                "pearson_with_heat_flow": pearson_heat,
                "spearman_with_heat_flow": spearman_heat,
                "strongest_original_feature": strongest_name,
                "strongest_abs_pearson_with_original": abs(strongest_value),
                "strongest_pearson_with_original": strongest_value,
            }
        )
    result = pd.DataFrame(rows)
    result["screening_note"] = np.where(
        result["strongest_abs_pearson_with_original"] >= 0.85,
        "Likely redundant with an existing feature",
        np.where(
            result["pearson_with_heat_flow"].abs() < 0.10,
            "Weak linear signal; retain only if core-area model comparison later supports it",
            "Candidate for a controlled core-area comparison",
        ),
    )
    return result.sort_values("pearson_with_heat_flow", key=lambda values: values.abs(), ascending=False)


def plot_heatmap(data, path):
    columns = ORIGINAL_FEATURES + SCS_CANDIDATES + [TARGET]
    correlation = data[columns].corr(method="pearson")
    fig, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=55, ha="right")
    axis.set_yticks(range(len(columns)), columns)
    for row in range(len(columns)):
        for col in range(len(columns)):
            value = correlation.iloc[row, col]
            if np.isfinite(value) and abs(value) >= 0.45:
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Core-area Pearson correlation: original and 3D-SCS features")
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_relationships(data, screening, path):
    selected = screening.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, feature in zip(axes.ravel(), selected):
        valid = data[[feature, TARGET]].dropna()
        axis.scatter(valid[feature], valid[TARGET], s=24, alpha=0.7, color="#2b6cb0")
        value = screening.loc[screening["feature"] == feature, "pearson_with_heat_flow"].iloc[0]
        axis.set_title(f"{feature}\nPearson r={value:.3f}")
        axis.set_xlabel(feature)
        axis.set_ylabel("heat_flow (mW/m2)")
    fig.suptitle("Core-area 3D-SCS feature relationships with heat flow", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)
    required = set(ORIGINAL_FEATURES + SCS_CANDIDATES + [TARGET, "scs_in_core_model"])
    absent = required - set(data.columns)
    if absent:
        raise ValueError(f"Input table is missing columns: {sorted(absent)}")

    core = data.loc[data["scs_in_core_model"]].copy()
    extended = data.copy()
    if len(core) < 30:
        raise ValueError(f"Only {len(core)} core-area records found; screening would be unreliable.")

    core_table = build_screening_table(core, "published_core_model")
    extended_table = build_screening_table(extended, "extended_grid_exploratory_only")
    all_table = pd.concat([core_table, extended_table], ignore_index=True)

    core_table.to_csv(OUTPUT_DIR / "3d_scs_screened_01_core_feature_screening.csv", index=False, float_format="%.5f")
    extended_table.to_csv(
        OUTPUT_DIR / "3d_scs_screened_02_extended_feature_screening_exploratory.csv",
        index=False,
        float_format="%.5f",
    )
    all_table.to_csv(OUTPUT_DIR / "3d_scs_screened_03_all_screening_results.csv", index=False, float_format="%.5f")
    core.to_csv(OUTPUT_DIR / "3d_scs_screened_04_core_stations_only.csv", index=False, float_format="%.5f")
    plot_heatmap(core, OUTPUT_DIR / "3d_scs_screened_05_core_correlation_heatmap.png")
    plot_relationships(core, core_table, OUTPUT_DIR / "3d_scs_screened_06_core_relationships.png")

    candidates = core_table.loc[
        (core_table["strongest_abs_pearson_with_original"] < 0.85)
        & (core_table["pearson_with_heat_flow"].abs() >= 0.10),
        "feature",
    ].tolist()
    summary = {
        "total_stations": int(len(data)),
        "published_core_stations": int(len(core)),
        "important_rule": (
            "Only the published-core results may justify a predictive experiment. "
            "Extended-grid results are descriptive and cannot by themselves support feature selection."
        ),
        "controlled_comparison_candidates": candidates,
        "core_screening_sorted_by_abs_pearson": core_table.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "3d_scs_screened_07_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("GFZ 3D-SCS feature screening complete")
    print("=" * 72)
    print(f"All stations: {len(data)}")
    print(f"Published-core stations: {len(core)}")
    print("\nCore-area results (ordered by absolute Pearson correlation with heat flow):")
    print(
        core_table[
            [
                "feature",
                "pearson_with_heat_flow",
                "spearman_with_heat_flow",
                "strongest_original_feature",
                "strongest_abs_pearson_with_original",
                "screening_note",
            ]
        ].to_string(index=False)
    )
    print("\nControlled-comparison candidates:", candidates if candidates else "none")
    print("\nSaved:")
    for filename in [
        "3d_scs_screened_01_core_feature_screening.csv",
        "3d_scs_screened_02_extended_feature_screening_exploratory.csv",
        "3d_scs_screened_03_all_screening_results.csv",
        "3d_scs_screened_04_core_stations_only.csv",
        "3d_scs_screened_05_core_correlation_heatmap.png",
        "3d_scs_screened_06_core_relationships.png",
        "3d_scs_screened_07_summary.json",
    ]:
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
