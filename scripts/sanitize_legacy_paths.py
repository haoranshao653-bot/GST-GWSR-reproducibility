"""Replace private desktop paths in archived text evidence with placeholders."""
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
USER_PATH = re.compile(r"C:(?:\\\\|\\)Users(?:\\\\|\\)[^\\\\\"']+")


def main() -> None:
    changed = 0
    for path in (ROOT / "results").rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf", ".svg", ".xlsx", ".docx"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new = text
        new = USER_PATH.sub("<local-user-path>", new)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    print(f"sanitized_files={changed}")


if __name__ == "__main__":
    main()
