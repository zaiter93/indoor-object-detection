"""Indoor object detection: dlib-XML -> leak-free splits -> YOLO11 -> COCO mAP."""

__version__ = "1.0.0"

# Bumped whenever notebooks/colab_train_eval.ipynb changes in a way that matters.
#
# Colab holds the notebook in the browser tab / Drive, not in the cloned repo, so
# "Disconnect and delete runtime" refreshes this code but leaves stale cells
# running against it. That mismatch silently produced a wrong submission twice.
# The notebook carries a matching constant and halts if the two disagree.
NOTEBOOK_REVISION = 2
