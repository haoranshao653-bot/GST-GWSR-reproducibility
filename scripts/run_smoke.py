"""Run a deliberately small end-to-end GST-GWSR smoke test."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "example" / "smoke_stations.csv"
OUT = ROOT / "run_outputs" / "smoke"
FEATURES = "topo,bouguer,sediment,moho,lab,cpd,lithos_km,volcano"


def main() -> int:
    command = [
        sys.executable, str(ROOT / "src" / "gwsr" / "gst_gwsr_core.py"),
        "--input", str(DATA), "--target", "heat_flow", "--lon", "lon", "--lat", "lat",
        "--features", FEATURES, "--output", str(OUT), "--k-neighbors", "5",
        "--population-size", "30", "--full-generations", "1", "--adapt-generations", "1",
        "--transfer-chains", "2", "--n-jobs", "1", "--max-query-points", "4",
    ]
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
