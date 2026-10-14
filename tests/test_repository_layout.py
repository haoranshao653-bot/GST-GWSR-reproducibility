from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_public_files_exist():
    for relative in ("README.md", "requirements.txt", "configs/final_outer_test.json", "data/README.md"):
        assert (ROOT / relative).exists(), relative


def test_required_schema_is_declared():
    text = (ROOT / "configs" / "final_outer_test.json").read_text(encoding="utf-8")
    for name in ("heat_flow", "topo", "bouguer", "volcano"):
        assert name in text
