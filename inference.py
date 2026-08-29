#!/usr/bin/env python3
"""Run the trained detector over a directory of images.

    python3 inference.py --input path/to/test_images --output path/to/predictions

Writes, into ``--output``:

``predictions.json``
    All detections in COCO format -- one flat list, machine-scoreable.
``<image_stem>.json``
    Per-image detections with absolute pixel boxes and class names.
``<image_stem>.jpg``
    The image with boxes drawn on it (disable with ``--no-images``).

Both a machine-readable and a visual form are produced because the brief does
not specify which is expected, and each is cheap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from indoor_det.config import CLASSES, WEIGHTS_DIR
from indoor_det.predict import CONF_DISPLAY, NMS_IOU, Detection, load_model, predict_images

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Distinct BGR colours per class, dark enough for white label text.
_COLOURS = [
    (178, 114, 0), (0, 159, 230), (115, 158, 0), (0, 86, 213),
    (167, 121, 204), (233, 180, 86), (66, 226, 240),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="directory of input images")
    parser.add_argument("--output", required=True, type=Path,
                        help="directory for predictions (created if absent)")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DIR / "best.pt",
                        help="trained checkpoint (default: weights/best.pt)")
    parser.add_argument("--conf", type=float, default=CONF_DISPLAY,
                        help=f"confidence threshold (default: {CONF_DISPLAY})")
    parser.add_argument("--iou", type=float, default=NMS_IOU,
                        help=f"NMS IoU threshold (default: {NMS_IOU})")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="inference image size; must match training for best results")
    parser.add_argument("--device", default=None,
                        help="'0' for GPU 0, 'cpu'. Default: auto-detect.")
    parser.add_argument("--batch", type=int, default=8, help="images per forward pass")
    parser.add_argument("--no-images", action="store_true",
                        help="write only JSON, skip annotated images")
    parser.add_argument("--recursive", action="store_true",
                        help="search --input recursively")
    return parser.parse_args()


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input is not a directory: {input_dir}")
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def draw(image_path: Path, detections: list[Detection], out_path: Path) -> None:
    """Render detections onto a copy of the image.

    Uses OpenCV, which arrives with Ultralytics -- no extra dependency.
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return
    thickness = max(2, round(0.002 * max(image.shape[:2])))

    for detection in detections:
        colour = _COLOURS[detection.class_id % len(_COLOURS)]
        x1, y1 = int(detection.x1), int(detection.y1)
        x2, y2 = int(detection.x2), int(detection.y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

        label = f"{CLASSES[detection.class_id]} {detection.score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Keep the label inside the frame when the box touches the top edge.
        top = max(y1, text_h + baseline + 2)
        cv2.rectangle(image, (x1, top - text_h - baseline - 2), (x1 + text_w + 4, top), colour, -1)
        cv2.putText(image, label, (x1 + 2, top - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), image)


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"error: weights not found at {args.weights}\n"
              f"Train first (python train.py) or pass --weights.", file=sys.stderr)
        return 1

    images = find_images(args.input, args.recursive)
    if not images:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[inference] {len(images)} images | weights={args.weights} | conf={args.conf}")

    model = load_model(args.weights, device=args.device)
    detections_by_file = predict_images(
        model, images,
        imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        device=args.device, batch=args.batch,
    )

    coco_predictions: list[dict] = []
    total = 0

    for index, image_path in enumerate(images, start=1):
        detections = detections_by_file.get(image_path.name, [])
        total += len(detections)

        per_image = {
            "image": image_path.name,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": CLASSES[d.class_id],
                    "confidence": round(d.score, 4),
                    "bbox_xyxy": [round(v, 2) for v in (d.x1, d.y1, d.x2, d.y2)],
                }
                for d in detections
            ],
        }
        (args.output / f"{image_path.stem}.json").write_text(
            json.dumps(per_image, indent=2), encoding="utf-8"
        )

        # image_id is the 1-based index into the sorted input listing, and the
        # mapping is recorded in predictions.json so it can be resolved back.
        for detection in detections:
            coco_predictions.append({
                "image_id": index,
                "file_name": image_path.name,
                "category_id": detection.class_id,
                "bbox": [round(v, 2) for v in detection.to_xywh()],
                "score": round(detection.score, 5),
            })

        if not args.no_images:
            draw(image_path, detections, args.output / f"{image_path.stem}.jpg")

    (args.output / "predictions.json").write_text(
        json.dumps(
            {
                "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)],
                "images": [{"id": i, "file_name": p.name}
                           for i, p in enumerate(images, start=1)],
                "annotations": coco_predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[inference] {total} detections across {len(images)} images -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
