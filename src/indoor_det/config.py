"""Project-wide constants and path resolution.

Kept dependency-free so that data parsing and analysis can run before the
heavier training stack (torch / ultralytics) is installed.
"""

from __future__ import annotations

from pathlib import Path

# Canonical class order. Fixed and alphabetical so that the integer id of a
# class never depends on iteration order of a dict or on which XML we read
# first -- label ids must stay stable across dataset rebuilds, otherwise a
# checkpoint trained yesterday silently mispredicts today.
CLASSES: tuple[str, ...] = (
    "chair",
    "clock",
    "exit",
    "fireextinguisher",
    "printer",
    "screen",
    "trashbin",
)
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Repo root = two levels up from this file (src/indoor_det/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"      # parsed annotations, split assignment
PROCESSED_DIR = DATA_DIR / "processed"  # YOLO-format dataset + COCO json
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
WEIGHTS_DIR = REPO_ROOT / "weights"
RUNS_DIR = REPO_ROOT / "runs"

SPLITS: tuple[str, ...] = ("train", "val", "test")

# Source images are a fixed 1280x720; verified across all six sequences. We
# still read the true size per image at parse time rather than trusting this,
# but it documents the expectation.
EXPECTED_SIZE = (1280, 720)


def find_raw_root() -> Path:
    """Locate the unmodified 'Indoor Object Detection Dataset' directory.

    Checked in preference order so the same code runs unchanged locally (where
    the dataset sits beside the repo) and on Colab (where it is extracted into
    data/raw/).
    """
    candidates = [
        DATA_DIR / "raw" / "Indoor Object Detection Dataset",
        REPO_ROOT / "Indoor Object Detection Dataset",
        DATA_DIR / "raw",
    ]
    for candidate in candidates:
        if (candidate / "annotation").is_dir():
            return candidate
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find the dataset. Looked for an 'annotation/' subdirectory in:\n  {searched}"
    )
