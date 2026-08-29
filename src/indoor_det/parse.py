"""Parse the dlib-XML annotations into a single validated, normalised record set.

The dataset ships one XML per video sequence, in dlib's box format: each
``<image>`` element carries a filename and zero or more ``<box>`` children
described by ``top`` / ``left`` / ``width`` / ``height`` plus a ``<label>``.

We convert to absolute ``[x1, y1, x2, y2]`` corners, clip to the image, drop
degenerate boxes, and attach the sequence id and frame index -- the frame index
is what makes the leak-free temporal split in :mod:`indoor_det.split` possible.

Deliberately stdlib-only (no Pillow/OpenCV): image dimensions are read straight
out of the JPEG SOF marker, which keeps this stage runnable in a bare
interpreter and costs one small function.
"""

from __future__ import annotations

import json
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CLASS_TO_ID, CLASSES, INTERIM_DIR, find_raw_root

FRAME_RE = re.compile(r"frame_s(\d+)_(\d+)\.jpg$", re.IGNORECASE)

# JPEG Start-Of-Frame markers that carry the image dimensions. C4/C8/CC are
# DHT/JPG/DAC respectively and must not be treated as SOF.
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def read_jpeg_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` by scanning JPEG segment markers."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"{path} is not a JPEG (missing SOI marker)")
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:  # resync: skip fill bytes / stray padding
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in _SOF_MARKERS:
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2  # standalone markers carry no length field
            continue
        i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
    raise ValueError(f"No SOF marker found in {path}")


@dataclass
class ImageRecord:
    """One annotated frame."""

    file: str            # path relative to the dataset root, POSIX separators
    sequence: int        # 1..6, the source video
    frame: int           # frame index within that sequence -- drives the split
    width: int
    height: int
    boxes: list[list[float]] = field(default_factory=list)  # [x1, y1, x2, y2]
    labels: list[int] = field(default_factory=list)         # class ids

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)


@dataclass
class ParseStats:
    """Book-keeping for the cleaning step, surfaced in the README."""

    images: int = 0
    boxes_kept: int = 0
    boxes_clipped: int = 0
    boxes_dropped_degenerate: int = 0
    empty_images: int = 0
    missing_files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.images} images, {self.boxes_kept} boxes kept "
            f"({self.boxes_clipped} clipped to bounds, "
            f"{self.boxes_dropped_degenerate} dropped as degenerate), "
            f"{self.empty_images} images with no objects, "
            f"{len(self.missing_files)} annotated files missing on disk"
        )


def _sequence_dir_for(xml_path: Path) -> str:
    """``annotation/annotation_s3.xml`` -> ``sequence_3``."""
    match = re.search(r"_s(\d+)\.xml$", xml_path.name)
    if not match:
        raise ValueError(f"Cannot infer sequence number from {xml_path.name}")
    return f"sequence_{int(match.group(1))}"


def parse_dataset(
    raw_root: Path | None = None,
    *,
    min_box_size: float = 2.0,
) -> tuple[list[ImageRecord], ParseStats]:
    """Parse every ``annotation_s*.xml`` into validated :class:`ImageRecord` objects.

    Args:
        raw_root: dataset root; auto-detected when omitted.
        min_box_size: boxes thinner than this (px, after clipping) are dropped.
            The dataset contains one 1px-wide box that is pure annotation noise
            and would otherwise become an impossible training target.
    """
    raw_root = Path(raw_root) if raw_root is not None else find_raw_root()
    xml_paths = sorted((raw_root / "annotation").glob("annotation_s*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No annotation_s*.xml files under {raw_root / 'annotation'}")

    records: list[ImageRecord] = []
    stats = ParseStats()

    for xml_path in xml_paths:
        seq_dir = _sequence_dir_for(xml_path)
        root = ET.parse(xml_path).getroot()

        for image_el in root.findall(".//image"):
            filename = image_el.get("file")
            if filename is None:
                continue
            # XMLs reference bare filenames; images live in the sequence folder.
            rel_path = f"{seq_dir}/{Path(filename).name}"
            abs_path = raw_root / rel_path
            if not abs_path.is_file():
                stats.missing_files.append(rel_path)
                continue

            match = FRAME_RE.search(Path(filename).name)
            if not match:
                raise ValueError(f"Unexpected frame filename: {filename}")
            sequence, frame = int(match.group(1)), int(match.group(2))

            width, height = read_jpeg_size(abs_path)
            record = ImageRecord(
                file=rel_path, sequence=sequence, frame=frame, width=width, height=height
            )

            for box_el in image_el.findall("box"):
                label_el = box_el.find("label")
                if label_el is None or label_el.text is None:
                    continue
                label = label_el.text.strip().lower()
                if label not in CLASS_TO_ID:
                    raise ValueError(f"Unknown label {label!r} in {xml_path.name}")

                top = float(box_el.get("top", 0))
                left = float(box_el.get("left", 0))
                box_w = float(box_el.get("width", 0))
                box_h = float(box_el.get("height", 0))
                x1, y1, x2, y2 = left, top, left + box_w, top + box_h

                # dlib annotations may extend past the frame when the annotator
                # dragged outside the canvas; clip rather than discard, since
                # the visible part is still a valid, learnable object.
                cx1, cy1 = max(0.0, x1), max(0.0, y1)
                cx2, cy2 = min(float(width), x2), min(float(height), y2)
                if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                    stats.boxes_clipped += 1

                if cx2 - cx1 < min_box_size or cy2 - cy1 < min_box_size:
                    stats.boxes_dropped_degenerate += 1
                    continue

                record.boxes.append([cx1, cy1, cx2, cy2])
                record.labels.append(CLASS_TO_ID[label])

            stats.boxes_kept += record.num_boxes
            # Object-free frames are kept on purpose: they are genuine negatives
            # (corridor with nothing annotated) and suppress false positives.
            if record.num_boxes == 0:
                stats.empty_images += 1
            records.append(record)
            stats.images += 1

    # Stable ordering: sequence, then frame. Everything downstream (splitting,
    # dataset export) depends on this being deterministic.
    records.sort(key=lambda r: (r.sequence, r.frame))
    return records, stats


def save_records(records: list[ImageRecord], path: Path | None = None) -> Path:
    """Serialise parsed records to JSON so later stages need not re-parse XML."""
    path = path or (INTERIM_DIR / "annotations.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"classes": list(CLASSES), "images": [asdict(r) for r in records]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_records(path: Path | None = None) -> list[ImageRecord]:
    """Inverse of :func:`save_records`."""
    path = path or (INTERIM_DIR / "annotations.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["classes"]) != CLASSES:
        raise ValueError("Class list in cached annotations differs from config.CLASSES")
    return [ImageRecord(**item) for item in payload["images"]]


if __name__ == "__main__":
    parsed, parse_stats = parse_dataset()
    print(parse_stats.summary())
    print("saved ->", save_records(parsed))
