"""Create a deterministic tiny input for import and schema smoke tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano"]


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "data" / "example" / "smoke_stations.csv"
    rng = np.random.default_rng(42)
    n = 32
    frame = pd.DataFrame({
        "lon": rng.uniform(105, 122, n),
        "lat": rng.uniform(0, 25, n),
        "heat_flow": rng.uniform(40, 180, n),
    })
    for index, name in enumerate(FEATURES):
        frame[name] = rng.normal(index, 1.0, n)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
