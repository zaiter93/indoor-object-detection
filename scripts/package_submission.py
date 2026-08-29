#!/usr/bin/env python3
"""Bundle the deliverables into submission.zip.

    python scripts/package_submission.py

Includes exactly what the brief asks for -- source, trained weights, the
``predictions/`` folder, ``requirements.txt`` and ``README.md`` -- plus the
metrics report and figures the README refers to. The raw dataset, training run
directories and caches are excluded: they are large, regenerable, and not part
of the submission.

The script verifies each required item is present and refuses to produce an
incomplete archive silently.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (path relative to repo root, required?)
INCLUDE: tuple[tuple[str, bool], ...] = (
    ("README.md", True),
    ("requirements.txt", True),
    ("inference.py", True),
    ("train.py", True),
    ("prepare_data.py", True),
    ("src", True),
    ("configs", True),
    ("scripts", True),
    ("notebooks", False),
    ("weights/best.pt", True),
    ("predictions", True),
    ("reports", False),
)

EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _iter_files(path: Path):
    """Yield files under ``path``, skipping caches."""
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if EXCLUDE_PARTS & set(item.parts) or item.suffix in EXCLUDE_SUFFIXES:
            continue
        yield item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "submission.zip")
    parser.add_argument("--allow-missing", action="store_true",
                        help="package anyway when required items are absent")
    args = parser.parse_args()

    missing = [
        name for name, required in INCLUDE
        if required and not (REPO_ROOT / name).exists()
    ]
    if missing and not args.allow_missing:
        print("error: required items are missing:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nRun `python train.py` (produces weights/best.pt and reports/) and\n"
            "`python inference.py --input <images> --output predictions` first,\n"
            "or pass --allow-missing.",
            file=sys.stderr,
        )
        return 1

    total_bytes = 0
    file_count = 0
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, _ in INCLUDE:
            source = REPO_ROOT / name
            if not source.exists():
                continue
            for file_path in _iter_files(source):
                archive.write(file_path, file_path.relative_to(REPO_ROOT).as_posix())
                total_bytes += file_path.stat().st_size
                file_count += 1

    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output}")
    print(f"  {file_count} files, {total_bytes / 1e6:.1f} MB uncompressed -> {size_mb:.1f} MB")
    if missing:
        print(f"  WARNING: packaged without {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
