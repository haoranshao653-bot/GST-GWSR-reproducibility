"""Bounded smoke test for the final exact GWSR engine."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "src" / "gwsr" / "exact"
sys.path.insert(0, str(SECTION))
import original_bisquare as base  # noqa: E402
import paper_gp  # noqa: E402


def main() -> int:
    frame = pd.read_csv(ROOT / "data" / "example" / "smoke_stations.csv")
    X = StandardScaler().fit_transform(frame[base.FEATURES].to_numpy())
    y = StandardScaler().fit_transform(np.log(frame[base.TARGET]).to_numpy().reshape(-1, 1)).ravel()
    weights = np.ones(len(frame), dtype=float) / len(frame)
    model = base.build_sr_model(0)
    model.set_params(population_size=20, generations=1, n_jobs=1)
    best, info = paper_gp.search(X, y, weights, seed=0, model=model)
    pred = float(best.execute(X[:1])[0])
    if not np.isfinite(pred):
        raise RuntimeError("smoke prediction is not finite")
    print(f"final_engine_smoke=pass; candidates={info['evaluated_candidates']}; prediction_scaled={pred:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
