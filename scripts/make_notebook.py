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

REPO_URL = "https://github.com/zaiter93/indoor-object-detection"
REVISION = 2  # keep in step with indoor_det.NOTEBOOK_REVISION
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
        "Clones the public repository. Change `REPO_URL` only if you are running "
        "from a fork.",
    ),
    code(
        "import os",
        "from pathlib import Path",
        "",
        f'REPO_URL = "{REPO_URL}"',
        'REPO_DIR = Path("/content/indoor-object-detection")',
        "",
        "# Bumped whenever these cells change; checked against the repo in the next",
        "# cell. Colab keeps the notebook in your browser tab, not in the cloned",
        "# repo, so stale cells can silently run against fresh code.",
        f"NOTEBOOK_REVISION = {REVISION}",
        "",
        "if not REPO_DIR.exists():",
        "    !git clone --depth 1 $REPO_URL {REPO_DIR}",
        "os.chdir(REPO_DIR)",
        "print('working directory:', Path.cwd())",
    ),
    code(
        "import sys",
        "sys.path.insert(0, 'src')",
        "from indoor_det import NOTEBOOK_REVISION as REPO_REVISION",
        "",
        "if NOTEBOOK_REVISION != REPO_REVISION:",
        "    print('=' * 72)",
        "    print(f'STALE NOTEBOOK: these cells are revision {NOTEBOOK_REVISION}, but the')",
        "    print(f'repository expects revision {REPO_REVISION}.')",
        "    print()",
        "    print('Deleting the runtime refreshes the cloned code but NOT this notebook:')",
        "    print('the cells live in your browser tab / Drive copy. Open the current')",
        "    print('notebook fresh from GitHub and run again:')",
        "    print('  https://colab.research.google.com/github/zaiter93/indoor-object-detection'",
        "          '/blob/master/notebooks/colab_train_eval.ipynb')",
        "    print('=' * 72)",
        "    raise SystemExit('stale notebook -- see the message above')",
        "",
        "print(f'notebook revision {NOTEBOOK_REVISION} matches the repository')",
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
        "The dataset is ~400 MB, too large for a normal git repository, so it is "
        "attached to a GitHub Release. `DATASET_URL` below already points at it — "
        "no setup needed, the next cell just downloads and unzips it.",
        "",
        "Clearing `DATASET_URL` falls back to mounting Google Drive, if you would "
        "rather supply your own copy.",
    ),
    code(
        "DATASET_URL = \"https://github.com/zaiter93/indoor-object-detection/releases/download/v1.0-dataset/indoor-object-detection-dataset.zip\"",
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
        "| epochs | 60, run to completion | Measured: a 120-epoch schedule early-stopped "
        "at 78 and scored 6 mAP@50-95 *lower*, having never reached the no-mosaic phase. |",
        "| mosaic | 1.0, off for last 10 epochs | Multiplies effective exposure to rare "
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
        "## 6 · Leakage ablation",
        "",
        "The training run above **is** the grouped arm. This cell trains the *same* "
        "configuration on the naive random split, so the only difference between the "
        "two numbers is the split itself.",
        "",
        "`--tag` marks it as a secondary run: outputs go to "
        "`weights/best_ablation_random.pt` and `reports/*_ablation_random.*`, so it "
        "cannot overwrite the model that `inference.py` and the submission ship.",
        "",
        "Set `RUN_ABLATION = False` to skip (saves roughly 45 minutes).",
    ),
    code(
        "RUN_ABLATION = True",
        "",
        "if RUN_ABLATION:",
        "    # Same config, same schedule, same seed -- only --dataset differs, so the",
        "    # comparison isolates the split. --tag keeps every output of this run",
        "    # separate from the primary model's.",
        "    !python train.py --config configs/default.yaml --dataset random --device 0 --tag ablation_random",
    ),
    code(
        "import json",
        "from pathlib import Path",
        "",
        "if RUN_ABLATION:",
        "    sources = [('grouped (leak-free)', 'reports/metrics.json'),",
        "               ('random (naive)', 'reports/metrics_ablation_random.json')]",
        "    rows = []",
        "    for label, path in sources:",
        "        if Path(path).exists():",
        "            m = json.loads(Path(path).read_text())['splits']['val']['overall']",
        "            rows.append((label, m['mAP@50'], m['mAP@50-95']))",
        "",
        "    print(f\"{'split':<22}{'mAP@50':>10}{'mAP@50-95':>12}\")",
        "    for label, ap50, ap in rows:",
        "        print(f'{label:<22}{ap50:>10.4f}{ap:>12.4f}')",
        "",
        "    if len(rows) == 2:",
        "        d50, d595 = rows[1][1] - rows[0][1], rows[1][2] - rows[0][2]",
        "        print()",
        "        print(f'The naive split reports {d50:+.4f} mAP@50 and {d595:+.4f} mAP@50-95 more')",
        "        print('than the leak-free split, for a model that is no better. That gap is')",
        "        print('memorised near-duplicate frames -- recall of images the model already')",
        "        print('saw, which would evaporate in a room it had not been trained on.')",
        "        print()",
        "        print('The gap is larger at the stricter IoU range -- the signature of the')",
        "        print('effect: having seen a near-identical frame helps most with placing the')",
        "        print('box precisely, which is exactly what mAP@50-95 rewards.')",
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


def validate(notebook: dict) -> None:
    """Check every code cell is syntactically valid Python before writing.

    A generated notebook is easy to break subtly -- an escape sequence that
    collapses into a real newline turns a string literal into a syntax error
    that only surfaces when the cell runs, an hour into a Colab session. Parsing
    each cell here costs nothing and catches it at generation time.

    IPython line magics (``!cmd``, ``%cmd``) are not valid Python, so they are
    replaced with ``pass`` at the same indentation before parsing.
    """
    import ast

    failures = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        neutralised = "\n".join(
            (line[: len(line) - len(line.lstrip())] + "pass")
            if line.lstrip().startswith(("!", "%"))
            else line
            for line in source.splitlines()
        )
        try:
            ast.parse(neutralised)
        except SyntaxError as error:
            failures.append(f"  cell {index}: {error}")

    if failures:
        raise SystemExit("Generated notebook has invalid cells:\n" + "\n".join(failures))


def main() -> None:
    validate(NOTEBOOK)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} ({len(CELLS)} cells, all code cells parse)")


if __name__ == "__main__":
    main()
