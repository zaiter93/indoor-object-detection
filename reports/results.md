# Results

Model: `yolo11n.pt`  |  split: `grouped`  |  train imgsz: 416  |  epochs: 2

Metrics are computed with `pycocotools` against COCO-format ground truth,
at confidence threshold 0.001 so the precision-recall curve is complete.

## val (200 images, 522 objects)

| class | instances | AP@50 | AP@50-95 |
|---|---:|---:|---:|
| chair | 188 | 0.0000 | 0.0000 |
| clock | 20 | 0.0000 | 0.0000 |
| exit | 57 | 0.0000 | 0.0000 |
| fireextinguisher | 207 | 0.0005 | 0.0003 |
| printer | 19 | 0.0132 | 0.0108 |
| screen | 15 | 0.0063 | 0.0057 |
| trashbin | 16 | 0.0415 | 0.0349 |
| **all (macro)** | **522** | **0.0088** | **0.0074** |

Whole-set mAP@50-95 **0.0074**, mAP@50 **0.0088**, mAP@75 0.0086. By object size -- medium 0.0000, large 0.0089.

## test (229 images, 390 objects)

| class | instances | AP@50 | AP@50-95 |
|---|---:|---:|---:|
| chair | 91 | 0.0000 | 0.0000 |
| clock | 25 | 0.0000 | 0.0000 |
| exit | 36 | 0.0000 | 0.0000 |
| fireextinguisher | 175 | 0.0024 | 0.0014 |
| printer | 19 | 0.0098 | 0.0087 |
| screen | 32 | 0.1759 | 0.1513 |
| trashbin | 12 | 0.0089 | 0.0058 |
| **all (macro)** | **390** | **0.0281** | **0.0239** |

Whole-set mAP@50-95 **0.0239**, mAP@50 **0.0281**, mAP@75 0.0267. By object size -- medium 0.0000, large 0.0241.

## Qualitative analysis

Scored at confidence >= 0.25; mean per-image F1 0.0.

| error type | count |
|---|---:|
| missed object (FN) | 522 |
| false positive (FP) | 0 |
|   of which wrong class | 0 |
|   of which loose box | 0 |
| true positive (TP) | 0 |

![good examples](figures/qualitative_good.png)

![bad examples](figures/qualitative_bad.png)
