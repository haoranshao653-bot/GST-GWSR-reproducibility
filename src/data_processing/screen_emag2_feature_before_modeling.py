"""Screen the EMAG2 sea-level magnetic anomaly before GW-SR modeling."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


DATA_DIR = Path(r"C:\Users\shen\Desktop\xinjiashuju")
INPUT_FILE = DATA_DIR / "emag2_station_features" / "heat_flow_stations_with_gebco_and_emag2.csv"
OUTPUT_DIR = DATA_DIR / "emag2_feature_screening"
TARGET = "heat_flow"
EMAG_FEATURE = "emag2_sealevel_anomaly_nt"
ORIGINAL_FEATURES = ["topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano"]


def save_heatmap(correlation: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 8.5))
    image = axis.imshow(correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.index)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(correlation.index, fontsize=9)
    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]
            if np.isfinite(value) and abs(value) >= 0.45:
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    axis.set_title("原始特征、EMAG2 磁异常与热流的 Pearson 相关性")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_relationship_plot(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    valid = frame.dropna(subset=[EMAG_FEATURE, TARGET, "bouguer", "moho"])

    axis = axes[0]
    points = axis.scatter(
        valid[EMAG_FEATURE], valid[TARGET], c=valid["bouguer"], cmap="viridis", s=20, alpha=0.75, edgecolors="none"
    )
    axis.set_xlabel("EMAG2 海平面磁异常 (nT)")
    axis.set_ylabel("热流 (mW/m²)")
    axis.set_title("磁异常与热流关系\n颜色表示 Bouguer 异常")
    axis.grid(alpha=0.25)
    colorbar = fig.colorbar(points, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("bouguer")

    axis = axes[1]
    axis.scatter(valid["moho"], valid[EMAG_FEATURE], color="#2c7fb8", s=20, alpha=0.7, edgecolors="none")
    axis.set_xlabel("moho")
    axis.set_ylabel("EMAG2 海平面磁异常 (nT)")
    axis.set_title("磁异常与 Moho 的关系")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def recommendation(abs_max_existing: float, heat_abs_pearson: float, heat_abs_spearman: float) -> str:
    if abs_max_existing >= 0.90:
        return "Exclude: strongly redundant with an existing predictor"
    if max(heat_abs_pearson, heat_abs_spearman) >= 0.15:
        return "Candidate: test as one additional predictor"
    return "Weak global relation: do not add to the main GW-SR yet"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input table not found: {INPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    required = set(ORIGINAL_FEATURES + [EMAG_FEATURE, TARGET, "emag2_valid"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Input table is missing columns: {missing}")

    usable = df[df["emag2_valid"]].copy()
    columns = ORIGINAL_FEATURES + [EMAG_FEATURE, TARGET]
    pearson = usable[columns].corr(method="pearson")
    spearman = usable[columns].corr(method="spearman")
    existing_pearson = pearson.loc[ORIGINAL_FEATURES, EMAG_FEATURE]
    existing_spearman = spearman.loc[ORIGINAL_FEATURES, EMAG_FEATURE]
    strongest_name = str(existing_pearson.abs().idxmax())
    strongest_value = float(existing_pearson.loc[strongest_name])
    heat_pearson = float(pearson.loc[EMAG_FEATURE, TARGET])
    heat_spearman = float(spearman.loc[EMAG_FEATURE, TARGET])
    max_abs_existing = float(existing_pearson.abs().max())
    decision = recommendation(max_abs_existing, abs(heat_pearson), abs(heat_spearman))

    summary_df = pd.DataFrame(
        [{
            "feature": EMAG_FEATURE,
            "valid_count": int(len(usable)),
            "missing_count": int(len(df) - len(usable)),
            "pearson_with_heat_flow": heat_pearson,
            "spearman_with_heat_flow": heat_spearman,
            "strongest_existing_feature_pearson": strongest_name,
            "strongest_existing_pearson": strongest_value,
            "max_abs_pearson_with_existing": max_abs_existing,
            "max_abs_spearman_with_existing": float(existing_spearman.abs().max()),
            "recommendation": decision,
        }]
    )

    stations_path = OUTPUT_DIR / "emag2_screened_01_stations.csv"
    pearson_path = OUTPUT_DIR / "emag2_screened_02_pearson_correlation.csv"
    spearman_path = OUTPUT_DIR / "emag2_screened_03_spearman_correlation.csv"
    summary_table_path = OUTPUT_DIR / "emag2_screened_04_feature_screening.csv"
    heatmap_path = OUTPUT_DIR / "emag2_screened_05_correlation_heatmap.png"
    relation_path = OUTPUT_DIR / "emag2_screened_06_relationships.png"
    summary_json_path = OUTPUT_DIR / "emag2_screened_07_summary.json"

    usable.to_csv(stations_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    pearson.to_csv(pearson_path, encoding="utf-8-sig", float_format="%.6f")
    spearman.to_csv(spearman_path, encoding="utf-8-sig", float_format="%.6f")
    summary_df.to_csv(summary_table_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    save_heatmap(pearson, heatmap_path)
    save_relationship_plot(usable, relation_path)

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_file": str(INPUT_FILE),
                "total_stations": int(len(df)),
                "valid_emag2_stations": int(len(usable)),
                "emag2_feature": EMAG_FEATURE,
                "pearson_with_heat_flow": heat_pearson,
                "spearman_with_heat_flow": heat_spearman,
                "strongest_existing_feature": strongest_name,
                "strongest_existing_feature_pearson": strongest_value,
                "recommendation": decision,
                "rule": "Exclude if absolute correlation with any original feature is >=0.90. Otherwise, use the RF/GW-SR comparison to judge incremental value.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("EMAG2 feature screening complete")
    print("=" * 72)
    print(f"Valid stations: {len(usable)}/{len(df)}")
    print(f"EMAG2 vs heat_flow: Pearson={heat_pearson:.4f}, Spearman={heat_spearman:.4f}")
    print(f"Strongest existing relation: {strongest_name}, Pearson={strongest_value:.4f}")
    print(f"Recommendation: {decision}")
    print("\nSaved:")
    for path in [stations_path, pearson_path, spearman_path, summary_table_path, heatmap_path, relation_path, summary_json_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
