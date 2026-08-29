"""Run a trained detector over images and emit detections in COCO format.

Shared by evaluation, visualisation and the ``inference.py`` CLI so that all
three see byte-identical model output.

One detail matters more than it looks: the confidence threshold differs by
purpose. Average Precision integrates precision over the *whole* recall range,
so scoring must keep essentially every detection (``CONF_EVAL = 0.001``);
thresholding at 0.25 first would truncate the PR curve and silently understate
mAP. Human-facing output has the opposite need -- a clean image -- so it uses
``CONF_DISPLAY``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Threshold used when computing mAP: keep nearly everything so the precision
# recall curve is complete.
CONF_EVAL = 0.001
# Threshold used for rendered images and the CLI's JSON output.
CONF_DISPLAY = 0.25
# NMS IoU threshold. 0.7 is the Ultralytics default and suits these scenes,
# where same-class objects (rows of chairs) genuinely overlap.
NMS_IOU = 0.7


@dataclass
class Detection:
    """One predicted box in absolute pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    def to_xywh(self) -> list[float]:
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]


def normalise_device(device: str | None) -> str | None:
    """Translate Ultralytics-style device strings into torch-style ones.

    Ultralytics accepts ``"0"`` and ``"0,1"`` for CUDA ordinals, but
    ``torch.nn.Module.to()`` rejects them. Callers pass whichever form is
    natural on the command line, so normalise here rather than at every call.
    """
    if device is None:
        return None
    device = str(device).strip()
    if device.isdigit():
        return f"cuda:{device}"
    if "," in device:
        # Multi-GPU: torch cannot .to() a device list, so leave the model on its
        # default device and let the trainer distribute it.
        return None
    return device


def load_model(weights: Path | str, device: str | None = None):
    """Load a trained Ultralytics model.

    Imported lazily so that the data-preparation modules stay importable
    without the training stack installed.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    torch_device = normalise_device(device)
    if torch_device is not None:
        model.to(torch_device)
    return model


def predict_images(
    model,
    image_paths: list[Path],
    *,
    imgsz: int = 960,
    conf: float = CONF_EVAL,
    iou: float = NMS_IOU,
    device: str | None = None,
    batch: int = 8,
    max_det: int = 300,
    verbose: bool = False,
) -> dict[str, list[Detection]]:
    """Run inference and return ``{filename: [Detection, ...]}``.

    Images are processed in batches rather than all at once so that memory use
    stays bounded regardless of how many images the caller passes.
    """
    results_by_file: dict[str, list[Detection]] = {}

    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start : start + batch]
        outputs = model.predict(
            [str(p) for p in chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            verbose=verbose,
        )
        for path, output in zip(chunk, outputs):
            detections: list[Detection] = []
            boxes = output.boxes
            if boxes is not None and len(boxes):
                # .cpu().numpy() once per image rather than per box.
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), score, class_id in zip(xyxy, scores, classes):
                    detections.append(
                        Detection(float(x1), float(y1), float(x2), float(y2),
                                  float(score), int(class_id))
                    )
            results_by_file[path.name] = detections

    return results_by_file


def to_coco_predictions(
    detections_by_file: dict[str, list[Detection]],
    gt_json: Path,
) -> list[dict]:
    """Convert per-file detections into a COCO detection list.

    Image ids are resolved through the ground-truth file, so a prediction for an
    image the ground truth does not contain is dropped rather than silently
    mis-attributed to the wrong id.
    """
    data = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    file_to_id = {image["file_name"]: image["id"] for image in data["images"]}

    coco_predictions: list[dict] = []
    for filename, detections in detections_by_file.items():
        image_id = file_to_id.get(filename)
        if image_id is None:
            continue
        for detection in detections:
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": detection.class_id,
                    "bbox": [round(v, 2) for v in detection.to_xywh()],
                    "score": round(detection.score, 5),
                }
            )
    return coco_predictions
