"""Write a SHA-256 manifest for the public-safe repository contents."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "run_outputs", "__pycache__"}
SKIP_NAMES = {"heat_flow_stations.csv"}


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        relative_parts = set(path.relative_to(ROOT).parts)
        if relative_parts & SKIP_PARTS:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    destination = ROOT / "MANIFEST.sha256"
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"manifest_entries={len(rows)}")


if __name__ == "__main__":
    main()
