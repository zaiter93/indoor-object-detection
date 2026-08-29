#!/usr/bin/env python3
"""Generate notebooks/colab_train_eval.ipynb.

The notebook is the submitted Colab deliverable. Generating it from source
keeps it reviewable in git (a hand-edited .ipynb diffs badly) and guarantees it
stays in step with the CLI flags the scripts actually accept.

    python scripts/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/YOUR_USERNAME/indoor-object-detection"
NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "colab_train_eval.ipynb"


def markdown(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(lines),
    }


def _source(lines: tuple[str, ...]) -> list[str]:
    """nbformat stores source as a list of lines, each ending in \\n except the last."""
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


CELLS = [
    markdown(
        "# Indoor Object Detection — YOLO11s",
        "",
        "Trains and evaluates a 7-class indoor object detector "
        "(`chair`, `clock`, `exit`, `fireextinguisher`, `printer`, `screen`, `trashbin`) "
        "on the TUT Indoor Object Detection Dataset.",
        "",
        "**What this notebook produces**",
        "",
        "1. Exploratory analysis of the 2,213 annotated frames.",
        "2. A leak-free 80/10/10 split with all seven classes in every split.",
        "3. A trained YOLO11s model.",
        "4. Per-class **and** whole-set validation mAP, computed with `pycocotools`.",
        "5. Good and bad qualitative examples from the validation set.",
        "6. A measured comparison against the naive random split, to show how much "
        "mAP that split inflates.",
        "",
        "**Runtime:** `Runtime → Change runtime type → GPU`. An **L4** (Colab Pro) "
        "completes the full run comfortably; a free **T4** also works but takes "
        "roughly twice as long.",
    ),
    markdown("## 1 · Environment"),
    code(
        "!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv",
        "",
        "import torch, platform",
        "print('python', platform.python_version())",
        "print('torch ', torch.__version__, '| cuda', torch.cuda.is_available())",
        "if torch.cuda.is_available():",
        "    print('device', torch.cuda.get_device_name(0))",
    ),
    markdown(
        "### Get the code",
        "",
        "Replace `REPO_URL` with your fork before running.",
    ),
    code(
        "import os",
        "from pathlib import Path",
        "",
        f'REPO_URL = "{REPO_URL}"',
        'REPO_DIR = Path("/content/indoor-object-detection")',
        "",
        "if not REPO_DIR.exists():",
        "    !git clone --depth 1 $REPO_URL {REPO_DIR}",
        "os.chdir(REPO_DIR)",
        "print('working directory:', Path.cwd())",
    ),
    code(
        "# Colab already ships a CUDA build of torch; installing it again wastes "
        "several minutes\n"
        "# and risks a version conflict, so only the remaining packages are installed.",
        "!pip install -q ultralytics pycocotools",
        "",
        "import ultralytics, pycocotools",
        "print('ultralytics', ultralytics.__version__)",
    ),
    markdown(
        "### Get the dataset",
        "",
        "The dataset is ~400 MB, which is too large for a normal git repository. "
        "Pick whichever route you set up:",
        "",
        "* **GitHub Release** (recommended — fully reproducible for a reviewer): "
        "attach `Indoor Object Detection Dataset.zip` to a release and paste the "
        "asset URL below.",
        "* **Google Drive**: upload the zip to your Drive and mount it.",
    ),
    code(
        "DATASET_URL = \"\"  # e.g. https://github.com/<user>/<repo>/releases/download/v1.0/dataset.zip",
        'RAW_DIR = Path("data/raw")',
        "RAW_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        'if not (RAW_DIR / "Indoor Object Detection Dataset" / "annotation").exists():',
        "    if DATASET_URL:",
        "        !wget -q --show-progress -O /content/dataset.zip $DATASET_URL",
        "    else:",
        "        from google.colab import drive",
        "        drive.mount('/content/drive')",
        "        # Adjust this path to wherever you put the zip in your Drive.",
        "        !cp '/content/drive/MyDrive/Indoor Object Detection Dataset.zip' /content/dataset.zip",
        "    !unzip -q /content/dataset.zip -d {RAW_DIR}",
        "",
        "!ls '{RAW_DIR}/Indoor Object Detection Dataset'",
    ),
    markdown(
        "## 2 · Data analysis and splitting",
        "",
        "`prepare_data.py` parses the dlib XML annotations, cleans them, runs the "
        "exploratory analysis, and materialises both splits in YOLO + COCO format.",
        "",
        "It fails loudly if any split is missing a class, so the brief's "
        "\"all classes in each split\" requirement is enforced, not assumed.",
    ),
    code("!python prepare_data.py"),
    markdown(
        "### Why not a random split?",
        "",
        "These are frames from six continuous video walk-throughs, not independent "
        "photographs. The figure below measures how quickly a frame stops "
        "resembling its neighbours: consecutive frames share a median box IoU of "
        "**0.68** (the same object, barely moved), falling to ~**0.03** by a gap "
        "of 10 frames.",
        "",
        "A random split leaves a median gap of **1 frame** between each validation "
        "image and its nearest training image — it is scored on pictures it has "
        "effectively already seen. The grouped split raises that to **12 frames**.",
    ),
    code(
        "from IPython.display import Image, display, Markdown",
        "",
        "for name in ['temporal_redundancy', 'class_distribution', 'per_sequence_heatmap',",
        "             'box_sizes', 'spatial_heatmap']:",
        "    display(Markdown(f'**{name}**'))",
        "    display(Image(f'reports/figures/{name}.png'))",
    ),
    markdown(
        "## 3 · Training",
        "",
        "All hyperparameters live in `configs/default.yaml`, each with the reasoning "
        "that produced it. The headline choices:",
        "",
        "| setting | value | why |",
        "|---|---|---|",
        "| model | `yolo11s` | 1.8k training frames from six rooms cannot support a "
        "larger backbone without overfitting; `n` leaves accuracy unused. |",
        "| imgsz | 960 | Frames are 1280×720. At 640 the smallest 5% of boxes shrink "
        "to ~27 px; 960 keeps them near 41 px. |",
        "| optimizer | AdamW, lr 1e-3, cosine | Converges in far fewer epochs than "
        "SGD on ~110 iterations/epoch with a pretrained backbone. |",
        "| mosaic | 1.0, off for last 15 epochs | Multiplies effective exposure to rare "
        "classes; disabled at the end so training finishes on the real distribution. |",
        "| flipud | 0.0 | These scenes are gravity-aligned — an upside-down exit sign "
        "cannot occur at inference time. |",
    ),
    code(
        "!python train.py --config configs/default.yaml --dataset grouped --device 0",
    ),
    markdown(
        "## 4 · Results",
        "",
        "Per-class and whole-set mAP, computed by `pycocotools` at confidence 0.001 "
        "so the precision–recall curve is not truncated.",
    ),
    code(
        "from IPython.display import Markdown, display",
        "display(Markdown(open('reports/results.md').read()))",
    ),
    markdown(
        "## 5 · Qualitative examples",
        "",
        "Chosen automatically by per-image F1 — best six and worst six — so the "
        "selection cannot be cherry-picked. Dashed white boxes are ground truth; "
        "solid coloured boxes are predictions.",
    ),
    code(
        "from IPython.display import Image, display, Markdown",
        "display(Markdown('### Good examples'))",
        "display(Image('reports/figures/qualitative_good.png'))",
        "display(Markdown('### Bad examples'))",
        "display(Image('reports/figures/qualitative_bad.png'))",
    ),
    markdown(
        "## 6 · Leakage ablation (optional)",
        "",
        "Trains the *same* configuration on the naive random split, at a shortened "
        "but identical schedule for both, and compares. The gap is the mAP that a "
        "random split hands you for free — and that would evaporate on a hidden "
        "test set.",
        "",
        "Set `RUN_ABLATION = False` to skip (saves roughly 40 minutes).",
    ),
    code(
        "RUN_ABLATION = True",
        "ABLATION_EPOCHS = 60  # identical for both splits, so the comparison is fair",
        "",
        "if RUN_ABLATION:",
        "    import yaml, copy",
        "    base = yaml.safe_load(open('configs/default.yaml'))",
        "    for split_name in ['grouped', 'random']:",
        "        cfg = copy.deepcopy(base)",
        "        cfg['epochs'] = ABLATION_EPOCHS",
        "        cfg['close_mosaic'] = 10",
        "        cfg['name'] = f'ablation_{split_name}'",
        "        path = f'configs/_ablation_{split_name}.yaml'",
        "        yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)",
        "        !python train.py --config {path} --dataset {split_name} --device 0",
        "        !cp reports/metrics.json reports/metrics_ablation_{split_name}.json",
    ),
    code(
        "import json",
        "from pathlib import Path",
        "",
        "if RUN_ABLATION:",
        "    rows = []",
        "    for split_name in ['grouped', 'random']:",
        "        p = Path(f'reports/metrics_ablation_{split_name}.json')",
        "        if p.exists():",
        "            m = json.loads(p.read_text())['splits']['val']['overall']",
        "            rows.append((split_name, m['mAP@50'], m['mAP@50-95']))",
        "    print(f\"{'split':<10}{'mAP@50':>10}{'mAP@50-95':>12}\")",
        "    for name, ap50, ap in rows:",
        "        print(f'{name:<10}{ap50:>10.4f}{ap:>12.4f}')",
        "    if len(rows) == 2:",
        "        d50 = rows[1][1] - rows[0][1]",
        "        print(f'\\nRandom split reports {d50:+.4f} mAP@50 versus the leak-free split.')",
        "        print('That difference is memorised near-duplicate frames, not detection skill.')",
    ),
    markdown(
        "## 7 · Inference CLI",
        "",
        "The deliverable CLI, run exactly as specified in the brief. It writes "
        "`predictions.json` (COCO format), a per-image JSON, and an annotated JPEG "
        "for every input image.",
    ),
    code(
        "!python inference.py --input data/processed/grouped/images/test --output predictions",
        "",
        "!ls predictions | head -10",
        "!echo '...' && ls predictions | wc -l",
    ),
    code(
        "import json",
        "from IPython.display import Image, display",
        "",
        "preds = json.load(open('predictions/predictions.json'))",
        "print(f\"{len(preds['annotations'])} detections over {len(preds['images'])} images\")",
        "",
        "# Show a few annotated outputs.",
        "from pathlib import Path",
        "for p in sorted(Path('predictions').glob('*.jpg'))[:3]:",
        "    display(Image(str(p), width=640))",
    ),
    markdown(
        "## 8 · Package the submission",
        "",
        "Bundles source, trained weights, predictions, metrics and figures into a "
        "single zip and downloads it.",
    ),
    code(
        "!python scripts/package_submission.py",
        "",
        "from google.colab import files",
        "files.download('submission.zip')",
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
