"""
Detect and remove heat-flow outliers before GW-SR modeling.

This script does not run GW-SR. It only:
1. Reads the original heat-flow station CSV.
2. Detects heat_flow outliers with IQR rules.
3. Saves removed points, cleaned data, and summary files.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


STATION_FILE = Path(
    r"C:\Users\shen\Desktop\hiyangshuju\qinxuelunwenfangfa\heat_flow\heat_flow_stations.csv"
)
OUTPUT_DIR = Path(r"C:\Users\shen\Desktop\GWSR修改")
RUN_NAME = "heatflow_outlier_iqr"
TARGET = "heat_flow"


def iqr_bounds(values: pd.Series, multiplier: float) -> tuple[float, float, float, float, float]:
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return q1, q3, iqr, lower, upper


def describe_values(values: pd.Series) -> dict[str, float]:
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(STATION_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    q1_15, q3_15, iqr_15, lower_15, upper_15 = iqr_bounds(df[TARGET], 1.5)
    q1_30, q3_30, iqr_30, lower_30, upper_30 = iqr_bounds(df[TARGET], 3.0)

    df = df.copy()
    df["outlier_1_5iqr"] = (df[TARGET] < lower_15) | (df[TARGET] > upper_15)
    df["outlier_3iqr"] = (df[TARGET] < lower_30) | (df[TARGET] > upper_30)

    removed = df[df["outlier_3iqr"]].copy()
    cleaned = df[~df["outlier_3iqr"]].copy()

    removed = removed.sort_values(TARGET, ascending=False)
    cleaned = cleaned.drop(columns=["outlier_1_5iqr", "outlier_3iqr"])

    summary = {
        "input_file": str(STATION_FILE),
        "target": TARGET,
        "original": describe_values(df[TARGET]),
        "iqr_1_5_rule": {
            "q1": q1_15,
            "q3": q3_15,
            "iqr": iqr_15,
            "lower_bound": lower_15,
            "upper_bound": upper_15,
            "outlier_count": int(df["outlier_1_5iqr"].sum()),
        },
        "iqr_3_rule_used_for_cleaning": {
            "q1": q1_30,
            "q3": q3_30,
            "iqr": iqr_30,
            "lower_bound": lower_30,
            "upper_bound": upper_30,
            "removed_count": int(df["outlier_3iqr"].sum()),
            "remaining_count": int(len(cleaned)),
        },
        "cleaned": describe_values(cleaned[TARGET]),
    }

    all_flagged_path = OUTPUT_DIR / f"{RUN_NAME}_01_all_points_with_flags.csv"
    removed_path = OUTPUT_DIR / f"{RUN_NAME}_02_removed_3iqr_points.csv"
    cleaned_path = OUTPUT_DIR / f"{RUN_NAME}_03_cleaned_stations_3iqr.csv"
    summary_path = OUTPUT_DIR / f"{RUN_NAME}_04_summary.json"

    df.to_csv(all_flagged_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    removed.to_csv(removed_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Heat-flow outlier detection only")
    print("=" * 70)
    print(f"Original count: {len(df)}")
    print(f"Original heat_flow: min={df[TARGET].min():.2f}, max={df[TARGET].max():.2f}")
    print()
    print("1.5*IQR diagnostic rule:")
    print(f"  bounds: [{lower_15:.2f}, {upper_15:.2f}]")
    print(f"  outliers: {int(df['outlier_1_5iqr'].sum())}")
    print()
    print("3*IQR conservative cleaning rule:")
    print(f"  bounds: [{lower_30:.2f}, {upper_30:.2f}]")
    print(f"  removed: {len(removed)}")
    print(f"  remaining: {len(cleaned)}")
    print()
    if len(removed):
        print("Removed points sorted by heat_flow:")
        cols = [col for col in ["lon", "lat", TARGET] if col in removed.columns]
        print(removed[cols].to_string(index=False))
    else:
        print("No points removed by the 3*IQR rule.")
    print()
    print("Saved:")
    print(f"  {all_flagged_path}")
    print(f"  {removed_path}")
    print(f"  {cleaned_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
