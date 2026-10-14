"""Print the versions required for a reproducibility record."""
from __future__ import annotations

import importlib.metadata as md
import platform
import sys

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "gplearn")

print(f"python={sys.version}")
print(f"platform={platform.platform()}")
for name in PACKAGES:
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=MISSING")
