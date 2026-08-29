"""Model-independent COCO evaluation: per-class and overall mAP.

Metrics are computed with ``pycocotools`` against the COCO ground truth written
by :mod:`indoor_det.build_dataset`, rather than read out of the training
framework's own validation loop. Three reasons:

1. It is the reference implementation the reported numbers are expected to be
   comparable to.
2. It scores predictions, not a model, so a second architecture can be dropped
   in and compared on exactly the same footing.
3. It keeps the reported metric honest -- decoupled from any framework-specific
   choice of confidence threshold or NMS setting made during training.

The brief asks for mAP on all seven classes individually *and* on the whole
validation set; :func:`evaluate_predictions` returns both.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .config import CLASSES

# COCOeval packs its 12 summary statistics into a fixed-order array.
_STAT_NAMES = (
    "mAP@50-95",
    "mAP@50",
    "mAP@75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def _per_class_ap(coco_eval: COCOeval, iou_index: int | None = None) -> dict[str, float]:
    """Extract per-category AP from a finished :class:`COCOeval`.

    ``coco_eval.eval["precision"]`` has shape
    ``[iou_thresholds, recall_thresholds, categories, area_ranges, max_dets]``.
    We take area range 0 ("all") and the last max-dets entry (100), average over
    recall thresholds, and either average over all 10 IoU thresholds
    (``iou_index=None``, giving AP@[.5:.95]) or select one (0 -> AP@50).

    Categories with no ground-truth instances yield -1 in this array; they are
    reported as NaN rather than silently averaged in as zero, which would
    understate the mean.
    """
    precision = coco_eval.eval["precision"]
    results: dict[str, float] = {}
    for class_id, name in enumerate(CLASSES):
        if iou_index is None:
            values = precision[:, :, class_id, 0, -1]
        else:
            values = precision[iou_index, :, class_id, 0, -1]
        values = values[values > -1]
        results[name] = float(np.mean(values)) if values.size else float("nan")
    return results


def evaluate_predictions(
    gt_json: Path,
    predictions: list[dict],
    *,
    verbose: bool = False,
) -> dict[str, object]:
    """Score detections against COCO ground truth.

    Args:
        gt_json: path to ``instances_<split>.json``.
        predictions: COCO-format detections, i.e. dicts with ``image_id``,
            ``category_id``, ``bbox`` (xywh) and ``score``.
        verbose: print COCOeval's own summary table.

    Returns:
        ``{"overall": {...}, "per_class": {...}, "num_predictions": int}``.
    """
    # pycocotools writes progress to stdout unconditionally; suppress unless asked.
    redirect = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())

    with redirect:
        coco_gt = COCO(str(gt_json))

        if not predictions:
            # loadRes raises on an empty list, but "the model predicted nothing"
            # is a legitimate outcome that should score 0, not crash.
            return {
                "overall": {name: 0.0 for name in _STAT_NAMES},
                "per_class": {
                    name: {"AP@50-95": 0.0, "AP@50": 0.0} for name in CLASSES
                },
                "num_predictions": 0,
            }

        coco_dt = coco_gt.loadRes(list(predictions))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    ap_all = _per_class_ap(coco_eval, None)
    ap_50 = _per_class_ap(coco_eval, 0)

    return {
        "overall": {name: float(value) for name, value in zip(_STAT_NAMES, coco_eval.stats)},
        "per_class": {
            name: {"AP@50-95": ap_all[name], "AP@50": ap_50[name]} for name in CLASSES
        },
        "num_predictions": len(predictions),
    }


def count_instances(gt_json: Path) -> dict[str, int]:
    """Ground-truth instance count per class, for context alongside each AP."""
    data = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    counts = {name: 0 for name in CLASSES}
    for annotation in data["annotations"]:
        counts[id_to_name[annotation["category_id"]]] += 1
    return counts


def format_metrics(metrics: dict[str, object], instances: dict[str, int] | None = None) -> str:
    """Render metrics as the per-class + overall table the brief asks for."""
    per_class = metrics["per_class"]  # type: ignore[index]
    overall = metrics["overall"]      # type: ignore[index]

    lines = [
        f"{'class':<18}{'instances':>11}{'AP@50':>10}{'AP@50-95':>11}",
        "-" * 50,
    ]
    for name in CLASSES:
        count = f"{instances[name]}" if instances else "-"
        row = per_class[name]
        lines.append(
            f"{name:<18}{count:>11}{row['AP@50']:>10.4f}{row['AP@50-95']:>11.4f}"
        )

    # Macro average over classes: every class counts equally, so the six rare
    # classes are not drowned out by chair/fireextinguisher. This is what
    # "mAP over all seven classes" means, and it matches COCO's own definition.
    lines.append("-" * 50)
    total = f"{sum(instances.values())}" if instances else "-"
    lines.append(
        f"{'ALL (macro)':<18}{total:>11}"
        f"{overall['mAP@50']:>10.4f}{overall['mAP@50-95']:>11.4f}"
    )
    lines.append("")
    lines.append(
        f"mAP@50-95 (whole val set): {overall['mAP@50-95']:.4f}   "
        f"mAP@50: {overall['mAP@50']:.4f}   mAP@75: {overall['mAP@75']:.4f}"
    )
    lines.append(
        f"by size -- medium: {overall['mAP_medium']:.4f}   large: {overall['mAP_large']:.4f}"
    )
    return "\n".join(lines)
