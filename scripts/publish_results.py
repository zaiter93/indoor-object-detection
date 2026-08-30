#!/usr/bin/env python3
"""Copy a finished run's results out of submission.zip and into the repository.

    python scripts/publish_results.py submission.zip

Colab's "Save in GitHub" commits only the notebook. Everything the training run
produced -- the checkpoint, the metrics, the report, the qualitative figures --
exists solely on the Colab VM and is lost when it recycles. The one copy that
reaches your machine is the downloaded ``submission.zip``, so that is what this
reads.

Refuses to publish results from a secondary (ablation) run, using the same check
``package_submission.py`` applies. That mistake shipped twice.

``predictions/`` is deliberately NOT published: 459 files and ~40 MB, already in
the zip that goes to the reviewer, and of no use inside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files lifted out of the archive into the repo. These are what the README
# links to, plus the checkpoint so a reviewer can run inference.py immediately.
PUBLISH = (
    "reports/results.md",
    "reports/metrics.json",
    "reports/figures/qualitative_good.png",
    "reports/figures/qualitative_bad.png",
    "weights/best.pt",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, nargs="?", default=REPO_ROOT / "submission.zip")
    parser.add_argument("--force", action="store_true",
                        help="publish even if the archive describes a secondary run")
    args = parser.parse_args()

    if not args.archive.exists():
        print(f"error: {args.archive} not found. Download submission.zip from Colab first.",
              file=sys.stderr)
        return 1

    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())

        # Same guard as package_submission: never publish the ablation's output
        # as though it were the reported model.
        if "reports/metrics.json" in names:
            metrics = json.loads(archive.read("reports/metrics.json").decode("utf-8"))
            dataset = metrics.get("dataset")
            if dataset != "grouped" and not args.force:
                print(
                    f"error: this archive describes the '{dataset}' split, not 'grouped'.\n"
                    "       A secondary run overwrote the primary results, so publishing it\n"
                    "       would put the wrong model's numbers in the repository.\n"
                    "       Re-run the notebook, or pass --force if you are certain.",
                    file=sys.stderr,
                )
                return 1
            overall = metrics["splits"]["val"]["overall"]
            print(f"archive: {dataset} split, {metrics['config']['epochs']} epochs, "
                  f"val mAP@50 {overall['mAP@50']:.4f}, mAP@50-95 {overall['mAP@50-95']:.4f}")

        missing = [name for name in PUBLISH if name not in names]
        if missing:
            print(f"error: archive is missing {missing}", file=sys.stderr)
            return 1

        for name in PUBLISH:
            target = REPO_ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            print(f"  published  {name}  ({target.stat().st_size / 1e6:.1f} MB)")

    print("\nNow commit them:")
    print("  git add reports weights && git commit -m 'Publish results from the Colab run'")
    print("  git push origin master")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
