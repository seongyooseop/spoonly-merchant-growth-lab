"""The analysis must not read the simulation's truth. Only src/validate.py may open data/truth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_FILES = ["src/analysis.py", "src/build_marts.py", "sql/marts.sql"]


def test_analysis_never_reads_truth():
    for f in ANALYSIS_FILES:
        text = (ROOT / f).read_text(encoding="utf-8").lower()
        assert "data/truth" not in text and "truth" not in text, f"{f} references the truth folder"


def test_world_constants_not_copied_into_analysis():
    # campaign weeks and the pilot rule must come from data/raw, not from world.py constants
    text = (ROOT / "src/analysis.py").read_text(encoding="utf-8")
    assert "from world import" not in text and "import world" not in text
    assert "[6, 7, 18, 19, 31, 32, 44, 45]" not in text


def test_validation_only_reader_of_truth():
    readers = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").glob("*.py")
               if "data\" / \"truth" in p.read_text(encoding="utf-8") or "\"truth\"" in p.read_text(encoding="utf-8")]
    assert set(readers) <= {"src/validate.py", "src/world.py"}, readers
