Trained weights land here.

- `best.pt`          — the primary model. This is what inference.py loads and what ships.
- `best_<tag>.pt`    — secondary runs (e.g. the leakage ablation), written when train.py
                        is given --tag. Never shipped, never used for inference.

Not committed: produced by the Colab run, then bundled into submission.zip.
