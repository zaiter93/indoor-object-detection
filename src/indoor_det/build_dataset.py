"""Materialise a split as an on-disk dataset: YOLO labels + COCO ground truth.

Two representations of the *same* split are emitted:

``images/`` + ``labels/`` + ``dataset.yaml``
    Ultralytics' expected layout, used for training.

``coco/instances_<split>.json``
    Standard COCO ground truth, used for evaluation via ``pycocotools``.

Emitting both is deliberate. Training and scoring then share one split
definition, and the reported mAP comes from the same reference implementation
the COCO leaderboard uses -- not from a framework's internal validation loop.
That makes the numbers comparable to published results and keeps the evaluator
independent of the model, so a second architecture can be scored identically.

Images are hard-linked rather than copied where the filesystem allows it, which
makes a rebuild near-instant and costs no extra disk.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import CLASSES, PROCESSED_DIR, SPLITS
from .parse import ImageRecord


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst``, falling back to a copy.

    Hard links fail across volumes and on some filesystems; a copy is always
    correct, just slower and larger.
    """
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _to_yolo_line(box: list[float], label: int, width: int, height: int) -> str:
    """``[x1,y1,x2,y2]`` -> ``"<cls> <cx> <cy> <w> <h>"`` normalised to [0,1]."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def build_coco(records: list[ImageRecord], indices: list[int]) -> dict:
    """Build a COCO-format ground-truth dict for the given records.

    Category ids are the same 0-based ids used everywhere else in this project
    (and by YOLO), rather than COCO's 1-based convention. Since both the ground
    truth and the predictions are produced here, consistency matters more than
    matching upstream COCO, and it removes a whole class of off-by-one bugs.
    """
    images, annotations = [], []
    annotation_id = 1
    for image_id, idx in enumerate(indices, start=1):
        record = records[idx]
        images.append(
            {
                "id": image_id,
                "file_name": Path(record.file).name,
                "width": record.width,
                "height": record.height,
            }
        )
        for box, label in zip(record.boxes, record.labels):
            x1, y1, x2, y2 = box
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO uses xywh
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
    }


def build(
    records: list[ImageRecord],
    split: dict[str, list[int]],
    raw_root: Path,
    out_dir: Path | None = None,
    *,
    name: str = "grouped",
) -> Path:
    """Write the full dataset for ``split`` and return the ``dataset.yaml`` path."""
    out_dir = Path(out_dir) if out_dir is not None else (PROCESSED_DIR / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coco").mkdir(exist_ok=True)

    for split_name in SPLITS:
        image_dir = out_dir / "images" / split_name
        label_dir = out_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for idx in split[split_name]:
            record = records[idx]
            stem = Path(record.file).stem
            _link_or_copy(raw_root / record.file, image_dir / f"{stem}.jpg")

            # An empty .txt is meaningful: it marks a true background image.
            # Omitting the file entirely would make Ultralytics log it as a
            # missing label instead of training on it as a negative.
            lines = [
                _to_yolo_line(box, label, record.width, record.height)
                for box, label in zip(record.boxes, record.labels)
            ]
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        coco_path = out_dir / "coco" / f"instances_{split_name}.json"
        coco_path.write_text(json.dumps(build_coco(records, split[split_name])), encoding="utf-8")

    yaml_path = out_dir / "dataset.yaml"
    class_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    yaml_path.write_text(
        "# Generated by indoor_det.build_dataset -- do not edit by hand.\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"{class_block}\n",
        encoding="utf-8",
    )
    return yaml_path
