#!/usr/bin/env python3
"""Verify the data pipeline's invariants.

    python scripts/run_checks.py

Deliberately not a pytest suite: these checks assert properties of the *built
dataset*, so they belong to the pipeline rather than to a unit-test framework,
and keeping them dependency-free means they run anywhere the project runs.

Every check either passes or raises with a message naming the exact violation.
Exit code 0 means the dataset on disk is safe to train on.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_det.config import CLASSES, NUM_CLASSES, PROCESSED_DIR, SPLITS
from indoor_det.parse import load_records, parse_dataset
from indoor_det.split import make_split, split_report, temporal_leakage

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str):
    """Decorator that records pass/fail instead of aborting on the first error."""
    def wrap(fn):
        try:
            detail = fn()
            PASSED.append(f"{name}{f'  ({detail})' if detail else ''}")
        except Exception as exc:  # noqa: BLE001 - we want every failure, not the first
            FAILED.append(f"{name}: {exc}")
        return fn
    return wrap


def main() -> int:
    records = load_records()

    @check("annotations parse to the expected shape")
    def _():
        parsed, stats = parse_dataset()
        assert len(parsed) == len(records), f"{len(parsed)} != cached {len(records)}"
        assert stats.images == 2213, f"expected 2213 images, got {stats.images}"
        assert not stats.missing_files, f"{len(stats.missing_files)} annotated files missing"
        return f"{stats.images} images, {stats.boxes_kept} boxes"

    @check("every box is inside its image and non-degenerate")
    def _():
        for record in records:
            for box in record.boxes:
                x1, y1, x2, y2 = box
                assert 0 <= x1 < x2 <= record.width, f"{record.file}: bad x {box}"
                assert 0 <= y1 < y2 <= record.height, f"{record.file}: bad y {box}"
        return "all 4594 boxes"

    @check("every label is a known class id")
    def _():
        for record in records:
            for label in record.labels:
                assert 0 <= label < NUM_CLASSES, f"{record.file}: label {label}"
        return f"{NUM_CLASSES} classes"

    @check("grouped split is deterministic")
    def _():
        a = make_split(records, strategy="grouped", block_size=40, seed=1)
        b = make_split(records, strategy="grouped", block_size=40, seed=1)
        assert a == b, "two identical calls produced different splits"
        return "same seed -> same split"

    for name in ("grouped", "random"):
        dataset_dir = PROCESSED_DIR / name
        if not dataset_dir.exists():
            FAILED.append(f"{name} dataset: not built (run prepare_data.py)")
            continue

        @check(f"[{name}] splits are disjoint and cover every image")
        def _(dataset_dir=dataset_dir):
            seen = Counter()
            for split in SPLITS:
                for path in (dataset_dir / "images" / split).glob("*.jpg"):
                    seen[path.name] += 1
            duplicated = [f for f, c in seen.items() if c > 1]
            assert not duplicated, f"{len(duplicated)} images in >1 split, e.g. {duplicated[:3]}"
            assert len(seen) == len(records), f"{len(seen)} images placed, expected {len(records)}"
            return f"{len(seen)} images, no overlap"

        @check(f"[{name}] every image has a label file")
        def _(dataset_dir=dataset_dir):
            for split in SPLITS:
                images = {p.stem for p in (dataset_dir / "images" / split).glob("*.jpg")}
                labels = {p.stem for p in (dataset_dir / "labels" / split).glob("*.txt")}
                assert images == labels, (
                    f"{split}: {len(images - labels)} images without labels, "
                    f"{len(labels - images)} labels without images"
                )
            return "1:1 in all splits"

        @check(f"[{name}] YOLO coordinates are normalised into [0, 1]")
        def _(dataset_dir=dataset_dir):
            count = 0
            for split in SPLITS:
                for path in (dataset_dir / "labels" / split).glob("*.txt"):
                    for line in path.read_text(encoding="utf-8").splitlines():
                        class_id, cx, cy, w, h = line.split()
                        assert 0 <= int(class_id) < NUM_CLASSES, f"{path}: class {class_id}"
                        for value in (float(cx), float(cy), float(w), float(h)):
                            assert 0.0 <= value <= 1.0, f"{path}: {line!r}"
                        count += 1
            return f"{count} boxes"

        @check(f"[{name}] all seven classes appear in all three splits")
        def _(dataset_dir=dataset_dir):
            for split in SPLITS:
                gt = json.loads(
                    (dataset_dir / "coco" / f"instances_{split}.json").read_text(encoding="utf-8")
                )
                present = {a["category_id"] for a in gt["annotations"]}
                missing = [CLASSES[i] for i in range(NUM_CLASSES) if i not in present]
                assert not missing, f"{split} missing {missing}"
            return "brief requirement satisfied"

        @check(f"[{name}] COCO ground truth is internally consistent")
        def _(dataset_dir=dataset_dir):
            for split in SPLITS:
                gt = json.loads(
                    (dataset_dir / "coco" / f"instances_{split}.json").read_text(encoding="utf-8")
                )
                image_ids = {img["id"] for img in gt["images"]}
                for annotation in gt["annotations"]:
                    assert annotation["image_id"] in image_ids, "annotation references unknown image"
                    _, _, w, h = annotation["bbox"]
                    assert w > 0 and h > 0, f"non-positive box {annotation['bbox']}"
            return "ids and boxes valid"

    @check("grouped split has materially less temporal leakage than random")
    def _():
        grouped = temporal_leakage(records, make_split(records, strategy="grouped", block_size=40, seed=1))
        naive = temporal_leakage(records, make_split(records, strategy="random"))
        assert grouped["val"] >= 10, f"grouped val gap {grouped['val']} < 10 frames"
        assert grouped["val"] > naive["val"], "grouped split is not better than random"
        return f"grouped val gap {grouped['val']:.0f} vs random {naive['val']:.0f} frames"

    @check("grouped split respects 80/10/10 within 2 points")
    def _():
        report = split_report(records, make_split(records, strategy="grouped", block_size=40, seed=1))
        targets = {"train": 0.8, "val": 0.1, "test": 0.1}
        worst = 0.0
        for split, target in targets.items():
            worst = max(worst, abs(report["splits"][split]["fraction"] - target))
        assert worst <= 0.02, f"largest deviation {worst:.3f} exceeds 0.02"
        return f"max deviation {worst * 100:.1f} points"

    for line in PASSED:
        print(f"  PASS  {line}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
