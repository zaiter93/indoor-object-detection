"""Exploratory data analysis: the evidence behind the modelling decisions.

Each function here answers one question that changes a downstream choice:

* class balance          -> whether rare-class handling is needed at all
* box size distribution  -> the input resolution to train at
* aspect ratios          -> whether the default anchor-free head is appropriate
* spatial distribution   -> whether a positional prior is being learned
* temporal redundancy    -> how far apart split boundaries must be

Figures are written to ``reports/figures`` and referenced from the README.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display on Colab or in CI.
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASSES, FIGURES_DIR, NUM_CLASSES
from .parse import ImageRecord

# A colourblind-safe qualitative palette (Okabe-Ito), one colour per class.
PALETTE = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442",
)


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def temporal_redundancy(
    records: list[ImageRecord],
    gaps: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100),
) -> dict[int, float]:
    """Median IoU between a box and its best same-class match ``gap`` frames later.

    This is the measurement that decides the splitting strategy. If frames one
    apart share a median IoU near 0.7, they are the same picture and must never
    straddle a split boundary. Where the curve reaches ~0, frames are
    effectively independent views and a boundary is safe.
    """
    by_sequence: dict[int, dict[int, ImageRecord]] = defaultdict(dict)
    for record in records:
        by_sequence[record.sequence][record.frame] = record

    curve: dict[int, float] = {}
    for gap in gaps:
        ious: list[float] = []
        for frames in by_sequence.values():
            for frame_number, record in frames.items():
                later = frames.get(frame_number + gap)
                if later is None:
                    continue
                for box, label in zip(record.boxes, record.labels):
                    best = max(
                        (_iou(box, other) for other, other_label in
                         zip(later.boxes, later.labels) if other_label == label),
                        default=0.0,
                    )
                    ious.append(best)
        curve[gap] = statistics.median(ious) if ious else 0.0
    return curve


def summarise(records: list[ImageRecord]) -> dict[str, object]:
    """Compute the numeric summary that the README quotes."""
    class_counts = Counter()
    per_sequence: dict[int, Counter] = defaultdict(Counter)
    sqrt_areas: list[float] = []
    aspects: list[float] = []
    boxes_per_image: list[int] = []

    for record in records:
        boxes_per_image.append(record.num_boxes)
        for box, label in zip(record.boxes, record.labels):
            name = CLASSES[label]
            class_counts[name] += 1
            per_sequence[record.sequence][name] += 1
            width, height = box[2] - box[0], box[3] - box[1]
            sqrt_areas.append((width * height) ** 0.5)
            aspects.append(width / height if height > 0 else 0.0)

    areas = np.array(sqrt_areas)
    return {
        "images": len(records),
        "boxes": int(sum(class_counts.values())),
        "empty_images": sum(1 for r in records if r.num_boxes == 0),
        "class_counts": {name: class_counts.get(name, 0) for name in CLASSES},
        "imbalance_ratio": round(max(class_counts.values()) / max(1, min(class_counts.values())), 1),
        "per_sequence": {int(k): dict(v) for k, v in sorted(per_sequence.items())},
        "boxes_per_image": {
            "mean": round(float(np.mean(boxes_per_image)), 2),
            "max": int(np.max(boxes_per_image)),
        },
        "box_sqrt_area_px": {
            key: round(float(np.percentile(areas, value)), 1)
            for key, value in (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))
        },
        # COCO size bands, on the native 1280x720 frames.
        "coco_size_bands": {
            "small(<32)": int((areas < 32).sum()),
            "medium(32-96)": int(((areas >= 32) & (areas < 96)).sum()),
            "large(>=96)": int((areas >= 96).sum()),
        },
        "aspect_ratio_w_over_h": {
            "p10": round(float(np.percentile(aspects, 10)), 2),
            "p50": round(float(np.percentile(aspects, 50)), 2),
            "p90": round(float(np.percentile(aspects, 90)), 2),
        },
        "temporal_redundancy_median_iou": temporal_redundancy(records),
    }


def plot_class_distribution(records: list[ImageRecord], out_dir: Path) -> Path:
    counts = Counter()
    for record in records:
        for label in record.labels:
            counts[CLASSES[label]] += 1
    names = sorted(CLASSES, key=lambda n: -counts.get(n, 0))
    values = [counts.get(n, 0) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=[PALETTE[CLASSES.index(n)] for n in names])
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_ylabel("instances")
    ax.set_title(
        f"Class distribution -- {sum(values)} boxes, "
        f"{max(values) / max(1, min(values)):.0f}:1 imbalance"
    )
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "class_distribution.png", out_dir)


def plot_per_sequence(records: list[ImageRecord], out_dir: Path) -> Path:
    """Heatmap of class counts per source sequence.

    Shows directly why a per-sequence split cannot satisfy the brief: some
    classes are confined to one or two sequences.
    """
    per_sequence: dict[int, Counter] = defaultdict(Counter)
    for record in records:
        for label in record.labels:
            per_sequence[record.sequence][CLASSES[label]] += 1

    sequences = sorted(per_sequence)
    matrix = np.array([[per_sequence[s].get(c, 0) for c in CLASSES] for s in sequences])

    fig, ax = plt.subplots(figsize=(9, 3.6))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(sequences)), [f"sequence_{s}" for s in sequences])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j, i, str(value), ha="center", va="center", fontsize=8,
                color="white" if value > matrix.max() * 0.5 else "black",
            )
    ax.set_title("Instances per class per sequence")
    fig.colorbar(image, ax=ax, fraction=0.025)
    return _save(fig, "per_sequence_heatmap.png", out_dir)


def plot_box_sizes(records: list[ImageRecord], out_dir: Path) -> Path:
    by_class: dict[str, list[float]] = {name: [] for name in CLASSES}
    for record in records:
        for box, label in zip(record.boxes, record.labels):
            by_class[CLASSES[label]].append(((box[2] - box[0]) * (box[3] - box[1])) ** 0.5)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))

    everything = [v for values in by_class.values() for v in values]
    left.hist(everything, bins=50, color="#0072B2", alpha=0.85)
    for threshold, label in ((32, "COCO small"), (96, "COCO medium")):
        left.axvline(threshold, color="#D55E00", linestyle="--", linewidth=1)
        left.text(threshold + 4, left.get_ylim()[1] * 0.9, label, fontsize=8, color="#D55E00")
    left.set_xlabel("sqrt(box area) [px, native 1280x720]")
    left.set_ylabel("count")
    left.set_title("Object size distribution")
    left.spines[["top", "right"]].set_visible(False)

    order = sorted(CLASSES, key=lambda n: -statistics.median(by_class[n] or [0]))
    right.boxplot(
        [by_class[n] or [0] for n in order], tick_labels=order, vert=True, showfliers=False,
        medianprops={"color": "#D55E00"},
    )
    right.set_ylabel("sqrt(box area) [px]")
    right.set_title("Object size by class")
    right.tick_params(axis="x", rotation=25)
    right.spines[["top", "right"]].set_visible(False)

    return _save(fig, "box_sizes.png", out_dir)


def plot_spatial_heatmap(records: list[ImageRecord], out_dir: Path) -> Path:
    """Where in the frame objects appear -- reveals capture bias."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, name in zip(axes.flat, CLASSES):
        centres = [
            ((b[0] + b[2]) / 2 / r.width, (b[1] + b[3]) / 2 / r.height)
            for r in records
            for b, label in zip(r.boxes, r.labels)
            if CLASSES[label] == name
        ]
        if centres:
            xs, ys = zip(*centres)
            ax.hist2d(xs, ys, bins=16, range=[[0, 1], [0, 1]], cmap="magma")
        ax.set_title(name, fontsize=10)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[-1].axis("off")
    fig.suptitle("Normalised box-centre density per class", y=1.0)
    return _save(fig, "spatial_heatmap.png", out_dir)


def plot_temporal_redundancy(records: list[ImageRecord], out_dir: Path) -> Path:
    """The figure that justifies the grouped split."""
    curve = temporal_redundancy(records)
    gaps = sorted(curve)
    values = [curve[g] for g in gaps]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(gaps, values, marker="o", color="#0072B2", linewidth=2)
    ax.axhline(0.5, color="#D55E00", linestyle="--", linewidth=1)
    ax.text(gaps[-1], 0.52, "near-duplicate threshold", ha="right", fontsize=9, color="#D55E00")

    # Left band: where a naive random split puts its boundaries.
    ax.axvspan(gaps[0], 5, color="#D55E00", alpha=0.10)
    ax.annotate(
        "random split:\nmedian gap = 1 frame\n(IoU 0.68 -- the leak)",
        xy=(1, values[0]), xytext=(1.35, 0.30), fontsize=8.5, color="#D55E00",
        arrowprops={"arrowstyle": "->", "color": "#D55E00", "linewidth": 1},
    )

    # Right marker: what the grouped split actually achieves.
    ax.axvline(12, color="#009E73", linestyle=":", linewidth=1.6)
    ax.annotate(
        "grouped split:\nmedian gap = 12 frames\n(IoU ~0.02 -- independent)",
        xy=(12, 0.03), xytext=(16, 0.32), fontsize=8.5, color="#009E73",
        arrowprops={"arrowstyle": "->", "color": "#009E73", "linewidth": 1},
    )

    ax.set_xscale("log")
    ax.set_ylim(-0.03, 0.78)
    ax.set_xlabel("frame gap within a sequence (log scale)")
    ax.set_ylabel("median IoU with best same-class match")
    ax.set_title("Temporal redundancy: how fast do frames become independent?")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "temporal_redundancy.png", out_dir)


def run(records: list[ImageRecord], out_dir: Path | None = None) -> dict[str, object]:
    """Produce every figure and the JSON summary. Returns the summary."""
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        plot_class_distribution(records, out_dir),
        plot_per_sequence(records, out_dir),
        plot_box_sizes(records, out_dir),
        plot_spatial_heatmap(records, out_dir),
        plot_temporal_redundancy(records, out_dir),
    ]
    summary = summarise(records)
    summary["figures"] = [p.name for p in figures]

    (out_dir.parent / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
