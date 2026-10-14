"""Validate the derived station table without modifying it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = [
    "lon", "lat", "heat_flow", "topo", "bouguer", "sediment", "moho",
    "lab", "cpd", "lithos_km", "volcano",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    missing = [column for column in REQUIRED if column not in frame.columns]
    numeric = frame.loc[:, [c for c in REQUIRED if c in frame.columns]].apply(pd.to_numeric, errors="coerce")
    nonfinite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if len(numeric.columns) else -1
    nonpositive_target = int((numeric["heat_flow"] <= 0).sum()) if "heat_flow" in numeric else -1
    report = {
        "input": str(args.input.resolve()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_required_columns": missing,
        "nonfinite_required_values": nonfinite,
        "nonpositive_heat_flow": nonpositive_target,
        "expected_station_count": 1024,
        "status": "PASS" if not missing and nonfinite == 0 and nonpositive_target == 0 else "FAIL",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
