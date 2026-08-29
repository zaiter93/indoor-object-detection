#!/usr/bin/env python3
"""Turn the raw dlib-XML dataset into trainable splits.

    python prepare_data.py

Steps: parse and clean the annotations, run the exploratory analysis, choose a
leak-free grouped split (and the naive random split used only as a baseline for
comparison), and write both to disk in YOLO + COCO form.

Deterministic: rerunning reproduces byte-identical splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from indoor_det.analyze import run as run_analysis
from indoor_det.build_dataset import build
from indoor_det.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, find_raw_root
from indoor_det.parse import parse_dataset, save_records
from indoor_det.split import (
    format_report,
    make_split,
    save_split,
    select_split,
    split_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=None,
                        help="dataset root; auto-detected when omitted")
    parser.add_argument("--datasets", nargs="+", default=["grouped", "random"],
                        choices=["grouped", "random"],
                        help="which splits to materialise (default: both)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="skip EDA figure generation")
    return parser.parse_args()


def prepare(
    raw_root: Path | None = None,
    datasets: tuple[str, ...] | list[str] = ("grouped", "random"),
    analyze: bool = True,
) -> dict[str, Path]:
    """Run the full preparation pipeline. Returns ``{name: dataset.yaml path}``."""
    raw_root = raw_root or find_raw_root()
    print(f"[prepare] dataset root: {raw_root}")

    records, stats = parse_dataset(raw_root)
    print(f"[prepare] {stats.summary()}")
    save_records(records)

    if analyze:
        print("[prepare] running exploratory analysis ...")
        summary = run_analysis(records)
        print(
            f"[prepare]   {summary['boxes']} boxes, imbalance {summary['imbalance_ratio']}:1, "
            f"median box {summary['box_sqrt_area_px']['p50']} px"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    built: dict[str, Path] = {}
    reports: dict[str, object] = {}

    for name in datasets:
        if name == "grouped":
            split, choice = select_split(records)
            print(f"[prepare] grouped split chosen: {json.dumps(choice)}")
            reports["grouped_selection"] = choice
        else:
            split = make_split(records, strategy="random")

        report = split_report(records, split)
        reports[name] = report
        print(f"\n[prepare] --- {name} split ---")
        print(format_report(report))
        print()

        # Fail loudly rather than train on a split that violates the brief.
        for split_name, row in report["splits"].items():  # type: ignore[union-attr]
            if row["missing_classes"]:
                raise AssertionError(
                    f"{name} split: {split_name} is missing {row['missing_classes']}"
                )

        save_split(split, records, INTERIM_DIR / f"split_{name}.json")
        built[name] = build(records, split, raw_root, name=name)
        print(f"[prepare] built -> {built[name]}")

    (REPORTS_DIR / "splits.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return built


def main() -> int:
    args = parse_args()
    prepare(raw_root=args.raw, datasets=args.datasets, analyze=not args.no_analysis)
    print(f"\n[prepare] done. Processed datasets under {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
