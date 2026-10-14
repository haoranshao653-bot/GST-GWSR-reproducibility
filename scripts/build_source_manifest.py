"""Record hashes for raw source and predictor-grid files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for path in sorted((ROOT / "data" / "raw").rglob("*")):
        if not path.is_file() or path.name == "source_manifest.json":
            continue
        rows.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    destination = ROOT / "data" / "raw" / "source_manifest.json"
    destination.write_text(json.dumps({"files": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"source_manifest_entries={len(rows)}")


if __name__ == "__main__":
    main()
