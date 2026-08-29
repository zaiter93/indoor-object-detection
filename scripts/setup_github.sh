#!/usr/bin/env bash
# Create the GitHub repository, push the code, and attach the dataset as a
# release asset.
#
#   bash scripts/setup_github.sh <repo-name> <public|private> [dataset.zip]
#
# Requires the GitHub CLI, authenticated:
#
#   winget install --id GitHub.cli -e
#   gh auth login          # interactive; needs a browser
#
# Safe to re-run: each step is skipped if it has already been done.

set -euo pipefail

REPO_NAME="${1:-indoor-object-detection}"
VISIBILITY="${2:-public}"
DATASET_ZIP="${3:-Indoor Object Detection Dataset.zip}"
RELEASE_TAG="v1.0-dataset"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "error: visibility must be 'public' or 'private', got '$VISIBILITY'" >&2
  exit 1
fi

command -v gh >/dev/null 2>&1 || {
  echo "error: gh not found. Install it with:  winget install --id GitHub.cli -e" >&2
  echo "       then restart the shell and run:  gh auth login" >&2
  exit 1
}

gh auth status >/dev/null 2>&1 || {
  echo "error: gh is installed but not authenticated. Run:  gh auth login" >&2
  exit 1
}

OWNER="$(gh api user --jq .login)"
echo "==> authenticated as $OWNER"

# --- 1. Repository ----------------------------------------------------------
if gh repo view "$OWNER/$REPO_NAME" >/dev/null 2>&1; then
  echo "==> repo $OWNER/$REPO_NAME already exists, reusing it"
else
  echo "==> creating $VISIBILITY repo $OWNER/$REPO_NAME"
  gh repo create "$REPO_NAME" "--$VISIBILITY" \
    --description "Indoor object detection (YOLO11s): leak-free video-aware splits, COCO evaluation, CLI inference."
fi

REMOTE_URL="https://github.com/$OWNER/$REPO_NAME.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi
echo "==> origin -> $REMOTE_URL"

# --- 2. Push ----------------------------------------------------------------
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "==> pushing branch '$BRANCH'"
git push -u origin "$BRANCH"

# --- 3. Dataset release -----------------------------------------------------
if [[ ! -f "$DATASET_ZIP" ]]; then
  echo "!! dataset zip not found at '$DATASET_ZIP' -- skipping the release." >&2
  echo "   Re-run with the path as the third argument once it is available." >&2
else
  # The asset name must not contain spaces: the notebook fetches it with wget,
  # and a spaced filename needs escaping at every call site.
  ASSET="indoor-object-detection-dataset.zip"
  cp -f "$DATASET_ZIP" "$ASSET"

  if gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
    echo "==> release $RELEASE_TAG exists; replacing the asset"
    gh release upload "$RELEASE_TAG" "$ASSET" --clobber
  else
    echo "==> creating release $RELEASE_TAG and uploading $(du -h "$ASSET" | cut -f1)"
    gh release create "$RELEASE_TAG" "$ASSET" \
      --title "Indoor Object Detection Dataset" \
      --notes "TUT Indoor Object Detection Dataset (v1.1, Bishwo Adhikari), attached so the Colab notebook can fetch it in one step. Original dataset belongs to its authors; redistributed here only to make this submission reproducible."
  fi
  rm -f "$ASSET"

  DATASET_URL="https://github.com/$OWNER/$REPO_NAME/releases/download/$RELEASE_TAG/$ASSET"
  echo "==> dataset URL: $DATASET_URL"
fi

# --- 4. Wire the notebook ---------------------------------------------------
echo
echo "Next:"
echo "  1. python scripts/make_notebook.py   # after setting REPO_URL in that file"
echo "  2. Open the notebook in Colab:"
echo "     https://colab.research.google.com/github/$OWNER/$REPO_NAME/blob/$BRANCH/notebooks/colab_train_eval.ipynb"
if [[ -n "${DATASET_URL:-}" ]]; then
  echo "  3. Paste this into the notebook's DATASET_URL cell:"
  echo "     $DATASET_URL"
fi
