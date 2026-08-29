#!/usr/bin/env python3
"""Train the indoor object detector, then score it with COCO metrics.

    python train.py --config configs/default.yaml

Runs end to end: builds the dataset if it is missing, trains, copies the best
checkpoint to ``weights/``, evaluates val and test with ``pycocotools``, renders
good/bad qualitative examples, and writes a Markdown report.

The training itself is delegated to Ultralytics; the value added here is that
every hyperparameter lives in a reviewable YAML file, and that the reported
metrics come from an independent evaluator rather than the trainer's own loop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml

from indoor_det.config import (
    CLASSES,
    FIGURES_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    REPO_ROOT,
    WEIGHTS_DIR,
)
from indoor_det.evaluate import count_instances, evaluate_predictions, format_metrics
from indoor_det.parse import ImageRecord, load_records
from indoor_det.predict import CONF_DISPLAY, CONF_EVAL, load_model, predict_images, to_coco_predictions
from indoor_det.visualize import render_examples

# Keys consumed by this script rather than passed through to Ultralytics.
_LOCAL_KEYS = {"model"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml",
                        help="training hyperparameters (YAML)")
    parser.add_argument("--data", type=Path, default=None,
                        help="dataset.yaml; defaults to data/processed/<dataset>/dataset.yaml")
    parser.add_argument("--dataset", default="grouped", choices=("grouped", "random"),
                        help="which prepared split to train on (default: grouped)")
    parser.add_argument("--device", default=None,
                        help="'0', 'cpu', ... Defaults to Ultralytics auto-detection.")
    parser.add_argument("--eval-imgsz", type=int, default=None,
                        help="inference size for evaluation; defaults to the training imgsz")
    parser.add_argument("--skip-eval", action="store_true",
                        help="train only, skip COCO evaluation and figures")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return config


def ensure_dataset(dataset: str) -> Path:
    """Return the dataset.yaml path, building the dataset if it is absent."""
    yaml_path = PROCESSED_DIR / dataset / "dataset.yaml"
    if yaml_path.exists():
        return yaml_path

    print(f"[train] {yaml_path} not found -- building it now.")
    from prepare_data import prepare  # local import: only needed on the cold path

    prepare(datasets=(dataset,), analyze=False)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset build did not produce {yaml_path}")
    return yaml_path


def evaluate_split(
    model,
    split: str,
    dataset_dir: Path,
    records_by_file: dict[str, ImageRecord],
    *,
    imgsz: int,
    device: str | None,
) -> tuple[dict, dict]:
    """Score one split and return ``(metrics, detections_by_file)``."""
    image_dir = dataset_dir / "images" / split
    gt_json = dataset_dir / "coco" / f"instances_{split}.json"
    image_paths = sorted(image_dir.glob("*.jpg"))

    # CONF_EVAL, not a display threshold: AP integrates the full recall range.
    detections = predict_images(
        model, image_paths, imgsz=imgsz, conf=CONF_EVAL, device=device
    )
    metrics = evaluate_predictions(gt_json, to_coco_predictions(detections, gt_json))
    metrics["instances"] = count_instances(gt_json)
    metrics["images"] = len(image_paths)
    return metrics, detections


def write_report(
    results: dict[str, dict],
    qualitative: dict | None,
    config: dict,
    dataset: str,
    out_path: Path,
) -> None:
    """Write the Markdown metrics report quoted by the README."""
    lines = [
        "# Results",
        "",
        f"Model: `{config.get('model')}`  |  split: `{dataset}`  |  "
        f"train imgsz: {config.get('imgsz')}  |  epochs: {config.get('epochs')}",
        "",
        "Metrics are computed with `pycocotools` against COCO-format ground truth,",
        "at confidence threshold 0.001 so the precision-recall curve is complete.",
        "",
    ]

    for split, metrics in results.items():
        overall = metrics["overall"]
        instances = metrics["instances"]
        lines += [
            f"## {split} ({metrics['images']} images, {sum(instances.values())} objects)",
            "",
            "| class | instances | AP@50 | AP@50-95 |",
            "|---|---:|---:|---:|",
        ]
        for name in CLASSES:
            row = metrics["per_class"][name]
            lines.append(
                f"| {name} | {instances[name]} | {row['AP@50']:.4f} | {row['AP@50-95']:.4f} |"
            )
        lines += [
            f"| **all (macro)** | **{sum(instances.values())}** | "
            f"**{overall['mAP@50']:.4f}** | **{overall['mAP@50-95']:.4f}** |",
            "",
            f"Whole-set mAP@50-95 **{overall['mAP@50-95']:.4f}**, "
            f"mAP@50 **{overall['mAP@50']:.4f}**, mAP@75 {overall['mAP@75']:.4f}. "
            f"By object size -- medium {overall['mAP_medium']:.4f}, large {overall['mAP_large']:.4f}.",
            "",
        ]

    if qualitative:
        lines += [
            "## Qualitative analysis",
            "",
            f"Scored at confidence >= {qualitative['confidence_threshold']}; "
            f"mean per-image F1 {qualitative['mean_f1']}.",
            "",
            "| error type | count |",
            "|---|---:|",
        ]
        for name, value in qualitative["error_totals"].items():
            lines.append(f"| {name} | {value} |")
        lines += [
            "",
            "![good examples](figures/qualitative_good.png)",
            "",
            "![bad examples](figures/qualitative_bad.png)",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_yaml = args.data or ensure_dataset(args.dataset)
    dataset_dir = data_yaml.parent

    train_kwargs = {k: v for k, v in config.items() if k not in _LOCAL_KEYS}
    train_kwargs["data"] = str(data_yaml)
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"[train] model={config['model']}  data={data_yaml}")
    # Load on CPU and let Ultralytics place the model: moving it to CUDA first
    # breaks its pretrained-class remapping, which copies COCO head weights for
    # names we share (chair, clock, tv -> screen) and is worth keeping.
    model = load_model(config["model"])
    results = model.train(**train_kwargs)

    # Ultralytics writes best.pt under <project>/<name>/weights/.
    best = Path(results.save_dir) / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    final_weights = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best, final_weights)
    print(f"[train] best checkpoint -> {final_weights}")

    if args.skip_eval:
        return 0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_imgsz = args.eval_imgsz or int(config.get("imgsz", 640))
    # Re-load from the saved checkpoint: this validates that the artefact we
    # ship is the artefact we scored.
    model = load_model(final_weights, device=args.device)

    records_by_file = {Path(r.file).name: r for r in load_records()}
    metrics_by_split: dict[str, dict] = {}
    val_detections: dict = {}

    for split in ("val", "test"):
        print(f"[eval] scoring {split} at imgsz={eval_imgsz} ...")
        metrics, detections = evaluate_split(
            model, split, dataset_dir, records_by_file,
            imgsz=eval_imgsz, device=args.device,
        )
        metrics_by_split[split] = metrics
        if split == "val":
            val_detections = detections
        print(f"\n--- {split} ---")
        print(format_metrics(metrics, metrics["instances"]))
        print()

    print("[eval] rendering qualitative examples ...")
    val_records = {
        name: record for name, record in records_by_file.items()
        if (dataset_dir / "images" / "val" / name).exists()
    }
    qualitative = render_examples(
        val_records, val_detections,
        image_dir=dataset_dir / "images" / "val",
        out_dir=FIGURES_DIR,
        conf=CONF_DISPLAY,
    )

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps({"config": config, "dataset": args.dataset,
                    "splits": metrics_by_split, "qualitative": qualitative}, indent=2),
        encoding="utf-8",
    )
    write_report(metrics_by_split, qualitative, config, args.dataset, REPORTS_DIR / "results.md")
    print(f"[eval] wrote {REPORTS_DIR / 'results.md'} and {REPORTS_DIR / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
