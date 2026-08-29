"""Qualitative results: automatically chosen good and bad validation examples.

Picking examples by hand invites cherry-picking. Instead every validation image
is scored by matching predictions to ground truth exactly the way the metric
does (greedy, highest-confidence-first, IoU >= 0.5, one prediction per object),
and the best and worst images by F1 are rendered.

Failures are also attributed to a cause -- missed object, false positive, wrong
class, or loose box -- because "mAP is 0.62" says nothing actionable, whereas
"most of the loss is missed printers at long range" does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from .analyze import PALETTE
from .config import CLASSES
from .parse import ImageRecord
from .predict import Detection

MATCH_IOU = 0.5
# Boxes that find the right class but land loosely (0.3 <= IoU < 0.5) are a
# different failure from a total miss, and are worth separating in the report.
LOOSE_IOU = 0.3


@dataclass
class ImageDiagnosis:
    """Per-image match result and error attribution."""

    filename: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    wrong_class: int = 0
    loose_box: int = 0
    matched: list[tuple[int, int]] = field(default_factory=list)  # (det index, gt index)

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        return (2 * self.true_positives / denominator) if denominator else 1.0

    @property
    def dominant_error(self) -> str:
        """The single biggest contributor to this image's error, for labelling."""
        candidates = {
            "missed object": self.false_negatives - self.wrong_class - self.loose_box,
            "false positive": self.false_positives - self.wrong_class,
            "wrong class": self.wrong_class,
            "loose box": self.loose_box,
        }
        cause, count = max(candidates.items(), key=lambda kv: kv[1])
        return cause if count > 0 else "clean"


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def diagnose(
    record: ImageRecord,
    detections: list[Detection],
    *,
    conf: float,
) -> ImageDiagnosis:
    """Match detections to ground truth and attribute every error."""
    kept = sorted([d for d in detections if d.score >= conf], key=lambda d: -d.score)
    diagnosis = ImageDiagnosis(filename=Path(record.file).name)

    gt_boxes = list(record.boxes)
    gt_labels = list(record.labels)
    gt_used = [False] * len(gt_boxes)

    for det_index, detection in enumerate(kept):
        det_box = [detection.x1, detection.y1, detection.x2, detection.y2]

        best_index, best_iou = -1, 0.0
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_used[gt_index]:
                continue
            overlap = _iou(det_box, gt_box)
            if overlap > best_iou:
                best_index, best_iou = gt_index, overlap

        if best_index >= 0 and best_iou >= MATCH_IOU and gt_labels[best_index] == detection.class_id:
            gt_used[best_index] = True
            diagnosis.true_positives += 1
            diagnosis.matched.append((det_index, best_index))
        else:
            diagnosis.false_positives += 1
            # Attribute the failure so the report can say *why*.
            if best_index >= 0 and best_iou >= MATCH_IOU:
                diagnosis.wrong_class += 1
            elif best_index >= 0 and best_iou >= LOOSE_IOU and gt_labels[best_index] == detection.class_id:
                diagnosis.loose_box += 1

    diagnosis.false_negatives = gt_used.count(False)
    return diagnosis


def _draw(ax, image_path: Path, record: ImageRecord, detections: list[Detection], conf: float) -> None:
    """Ground truth as dashed white outlines; predictions solid, per-class colour."""
    ax.imshow(Image.open(image_path).convert("RGB"))

    for box, label in zip(record.boxes, record.labels):
        ax.add_patch(
            patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor="white", linewidth=2.2, linestyle="--",
            )
        )
        ax.text(
            box[0] + 3, box[1] + 16, CLASSES[label], fontsize=7, color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
        )

    for detection in detections:
        if detection.score < conf:
            continue
        colour = PALETTE[detection.class_id % len(PALETTE)]
        ax.add_patch(
            patches.Rectangle(
                (detection.x1, detection.y1),
                detection.x2 - detection.x1, detection.y2 - detection.y1,
                fill=False, edgecolor=colour, linewidth=2.0,
            )
        )
        ax.text(
            detection.x1 + 3, detection.y2 - 6,
            f"{CLASSES[detection.class_id]} {detection.score:.2f}",
            fontsize=7, color="black",
            bbox={"facecolor": colour, "alpha": 0.85, "pad": 1, "edgecolor": "none"},
        )

    ax.set_xticks([])
    ax.set_yticks([])


def _select_diverse(
    ranked: list[ImageDiagnosis],
    records_by_file: dict[str, ImageRecord],
    count: int,
    *,
    min_frame_separation: int = 15,
) -> list[ImageDiagnosis]:
    """Take the top ``count`` entries, but never two near-duplicate frames.

    Ranking purely by F1 tends to return six consecutive frames of one scene,
    because consecutive frames fail in the same way. That fills the figure with
    one failure mode and hides the rest. Skipping images within
    ``min_frame_separation`` frames of an already-chosen one keeps the ordering
    (still worst-first) while showing distinct scenes.

    Falls back to filling from the remaining candidates if the constraint is too
    strict to reach ``count``.
    """
    chosen: list[ImageDiagnosis] = []
    taken: list[tuple[int, int]] = []  # (sequence, frame) of chosen images

    for diagnosis in ranked:
        record = records_by_file[diagnosis.filename]
        too_close = any(
            sequence == record.sequence and abs(frame - record.frame) < min_frame_separation
            for sequence, frame in taken
        )
        if too_close:
            continue
        chosen.append(diagnosis)
        taken.append((record.sequence, record.frame))
        if len(chosen) == count:
            return chosen

    # Not enough distinct scenes: top up in rank order rather than return short.
    for diagnosis in ranked:
        if len(chosen) == count:
            break
        if diagnosis not in chosen:
            chosen.append(diagnosis)
    return chosen


def render_examples(
    records_by_file: dict[str, ImageRecord],
    detections_by_file: dict[str, list[Detection]],
    image_dir: Path,
    out_dir: Path,
    *,
    conf: float = 0.25,
    count: int = 6,
) -> dict[str, object]:
    """Render the best and worst validation images and summarise error causes.

    Returns a summary dict (also useful for the README) describing what was
    rendered and how the failures break down.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnoses = [
        diagnose(record, detections_by_file.get(filename, []), conf=conf)
        for filename, record in records_by_file.items()
    ]
    # Only images that actually contain objects are informative as "good"
    # examples; an empty image with no predictions trivially scores F1 = 1.
    with_objects = [d for d in diagnoses if records_by_file[d.filename].num_boxes > 0]

    best_first = sorted(with_objects, key=lambda d: (-d.f1, d.false_positives + d.false_negatives))
    worst_first = sorted(with_objects, key=lambda d: (d.f1, -(d.false_positives + d.false_negatives)))
    good = _select_diverse(best_first, records_by_file, count)
    bad = _select_diverse(worst_first, records_by_file, count)

    for title, group, filename in (
        ("Good examples -- highest F1 on the validation set", good, "qualitative_good.png"),
        ("Bad examples -- lowest F1 on the validation set", bad, "qualitative_bad.png"),
    ):
        columns = 3
        rows = (len(group) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 5.2, rows * 3.1))
        for ax, diagnosis in zip(axes.flat, group):
            record = records_by_file[diagnosis.filename]
            _draw(ax, image_dir / diagnosis.filename, record,
                  detections_by_file.get(diagnosis.filename, []), conf)
            ax.set_title(
                f"{diagnosis.filename}\n"
                f"F1={diagnosis.f1:.2f}  TP={diagnosis.true_positives} "
                f"FP={diagnosis.false_positives} FN={diagnosis.false_negatives}"
                + (f"  [{diagnosis.dominant_error}]" if diagnosis.f1 < 1.0 else ""),
                fontsize=8,
            )
        for ax in axes.flat[len(group):]:
            ax.axis("off")
        fig.suptitle(
            f"{title}\n"
            "dashed white = ground truth, solid colour = prediction "
            f"(conf >= {conf})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=140, bbox_inches="tight")
        plt.close(fig)

    error_totals = {
        "missed object (FN)": sum(d.false_negatives for d in diagnoses),
        "false positive (FP)": sum(d.false_positives for d in diagnoses),
        "  of which wrong class": sum(d.wrong_class for d in diagnoses),
        "  of which loose box": sum(d.loose_box for d in diagnoses),
        "true positive (TP)": sum(d.true_positives for d in diagnoses),
    }
    return {
        "confidence_threshold": conf,
        "images_scored": len(diagnoses),
        "mean_f1": round(sum(d.f1 for d in with_objects) / max(1, len(with_objects)), 4),
        "error_totals": error_totals,
        "good_examples": [
            {"file": d.filename, "f1": round(d.f1, 3)} for d in good
        ],
        "bad_examples": [
            {"file": d.filename, "f1": round(d.f1, 3), "cause": d.dominant_error}
            for d in bad
        ],
    }
