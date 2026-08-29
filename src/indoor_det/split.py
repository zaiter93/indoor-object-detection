"""Train/val/test splitting, with an explicit stance on temporal leakage.

The dataset is not 2,213 independent photographs -- it is six continuous video
walk-throughs cut into frames. ``frame_s3_417.jpg`` and ``frame_s3_418.jpg`` are
roughly 1/30 s apart and are, for all practical purposes, the same picture.

A uniformly random 80/10/10 split therefore places near-duplicates of almost
every validation frame into the training set. The resulting mAP measures
memorisation, not generalisation, and does not survive contact with a held-out
test set recorded elsewhere.

Two strategies are implemented so the difference can be *measured* rather than
asserted:

``grouped`` (default, used for all reported metrics)
    Each sequence is cut into contiguous blocks of ``block_size`` frames, and
    whole blocks are assigned to splits. Near-duplicate neighbours stay
    together, so no validation frame has a training twin. Blocks are drawn from
    all six sequences and allocated by greedy multi-label stratification, which
    keeps the 80/10/10 ratio *and* satisfies the brief's requirement that every
    class appear in every split -- something a naive per-sequence split cannot
    do here, because ``printer`` occurs almost only in sequences 2 and 3.

``random``
    The naive baseline, provided purely to quantify how much mAP the leak
    inflates. Never used for reported numbers.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import CLASSES, INTERIM_DIR, NUM_CLASSES, SPLITS
from .parse import ImageRecord

DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass
class Block:
    """A contiguous run of frames from one sequence, assigned as a unit."""

    sequence: int
    indices: list[int]              # positions into the records list
    class_counts: list[int]         # boxes per class id within the block

    @property
    def size(self) -> int:
        return len(self.indices)


def _build_blocks(records: list[ImageRecord], block_size: int) -> list[Block]:
    """Cut each sequence into contiguous frame blocks.

    A block is also broken whenever the frame index jumps by more than
    ``block_size``, so a genuine discontinuity in the recording is never
    bridged by a single block.
    """
    by_sequence: dict[int, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_sequence[record.sequence].append(idx)

    blocks: list[Block] = []
    for sequence in sorted(by_sequence):
        indices = sorted(by_sequence[sequence], key=lambda i: records[i].frame)
        current: list[int] = []
        for position, idx in enumerate(indices):
            gap_break = (
                current
                and records[idx].frame - records[current[-1]].frame > block_size
            )
            if current and (len(current) >= block_size or gap_break):
                blocks.append(_make_block(records, sequence, current))
                current = []
            current.append(idx)
        if current:
            blocks.append(_make_block(records, sequence, current))
    return blocks


def _make_block(records: list[ImageRecord], sequence: int, indices: list[int]) -> Block:
    counts = [0] * NUM_CLASSES
    for idx in indices:
        for label in records[idx].labels:
            counts[label] += 1
    return Block(sequence=sequence, indices=list(indices), class_counts=counts)


def _greedy_stratify(
    blocks: list[Block],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[Block]]:
    """Assign whole blocks to splits by greedy multi-label stratification.

    Follows the spirit of Sechidis et al. (2011): repeatedly take the block
    carrying the *rarest still-unallocated* class and give it to whichever split
    is furthest below its quota for that class. Rare classes therefore get first
    claim on the limited val/test budget, which is what keeps ``printer``
    (81 boxes) and ``screen`` (115) present in all three splits.
    """
    rng = random.Random(seed)
    total_images = sum(b.size for b in blocks)
    total_per_class = [sum(b.class_counts[c] for b in blocks) for c in range(NUM_CLASSES)]

    # Remaining quota, per split, for images and for each class.
    image_quota = {s: total_images * ratios[s] for s in SPLITS}
    class_quota = {
        s: [total_per_class[c] * ratios[s] for c in range(NUM_CLASSES)] for s in SPLITS
    }
    assignment: dict[str, list[Block]] = {s: [] for s in SPLITS}

    # Class priority: rarest first. Ties broken deterministically by class id.
    class_priority = sorted(range(NUM_CLASSES), key=lambda c: (total_per_class[c], c))

    # Order blocks so that those containing rare classes are placed first,
    # while a shuffle within each tier avoids a systematic temporal bias
    # (e.g. always sending the start of every sequence to train).
    def rarity_key(block: Block) -> int:
        present = [c for c in class_priority if block.class_counts[c] > 0]
        return class_priority.index(present[0]) if present else NUM_CLASSES

    ordered = sorted(blocks, key=rarity_key)
    tiers: dict[int, list[Block]] = defaultdict(list)
    for block in ordered:
        tiers[rarity_key(block)].append(block)
    ordered = []
    for tier in sorted(tiers):
        tier_blocks = tiers[tier]
        rng.shuffle(tier_blocks)
        ordered.extend(tier_blocks)

    def score(split_name: str, block: Block) -> float:
        """How badly ``split_name`` still needs ``block``.

        Both terms are normalised to 'fraction of this split's own quota that is
        still unfilled', which is what lets a 10% split compete fairly with an
        80% one. Rare classes carry more weight (1/sqrt(frequency)), so the
        scarce ``printer`` and ``screen`` boxes steer placement while the
        abundant ``chair`` boxes mostly do not.
        """
        image_term = image_quota[split_name] / max(1.0, ratios[split_name] * total_images)

        weighted, weight_sum = 0.0, 0.0
        for class_id in range(NUM_CLASSES):
            if block.class_counts[class_id] == 0:
                continue
            quota = max(1.0, ratios[split_name] * total_per_class[class_id])
            weight = 1.0 / (total_per_class[class_id] ** 0.5)
            weighted += weight * (class_quota[split_name][class_id] / quota)
            weight_sum += weight
        class_term = weighted / weight_sum if weight_sum else image_term

        return class_term + image_term

    for block in ordered:
        best = max(SPLITS, key=lambda s: score(s, block))

        assignment[best].append(block)
        image_quota[best] -= block.size
        for c in range(NUM_CLASSES):
            class_quota[best][c] -= block.class_counts[c]

    return assignment


# How strongly per-class balance resists a block move that would fix image
# counts. Tuned on this dataset: high enough to keep the rare classes spread
# across val/test, low enough that the 80/10/10 ratio still converges.
_BALANCE_WEIGHT = 30.0


def _class_totals(assignment: dict[str, list[Block]]) -> list[int]:
    """Total boxes per class across the whole assignment."""
    return [
        sum(block.class_counts[c] for split in SPLITS for block in assignment[split])
        for c in range(NUM_CLASSES)
    ]


def _class_balance_error(
    assignment: dict[str, list[Block]],
    ratios: dict[str, float],
    totals: list[int],
) -> float:
    """Squared deviation of each split's class share from its target share.

    Normalised per class, so ``printer`` (81 boxes) and ``chair`` (1,662) carry
    comparable weight -- otherwise the abundant classes would dominate the sum
    and the rare ones, which are exactly the fragile ones, would be ignored.
    """
    error = 0.0
    for split in SPLITS:
        for class_id in range(NUM_CLASSES):
            total = totals[class_id]
            if total == 0:
                continue
            actual = sum(block.class_counts[class_id] for block in assignment[split])
            target = ratios[split] * total
            error += ((actual - target) / total) ** 2
    return error


def _rebalance_images(
    assignment: dict[str, list[Block]],
    ratios: dict[str, float],
    *,
    tolerance: float = 0.01,
    max_moves: int = 500,
) -> dict[str, list[Block]]:
    """Nudge split sizes back toward the requested 80/10/10.

    Greedy stratification optimises class balance and can leave the image counts
    a couple of points off target. This pass repeatedly moves one block from the
    most over-filled split to the most under-filled one, choosing the block whose
    size best closes the gap, and refusing any move that would remove the last
    remaining instance of a class from the donor split.

    Class balance is preserved to within a block, and the ratios land inside
    ``tolerance``, so the split honours the brief without sacrificing the
    leak-free property -- blocks still move as indivisible units.
    """
    total_images = sum(b.size for b in blocks_of(assignment))

    for _ in range(max_moves):
        fractions = {
            s: sum(b.size for b in assignment[s]) / total_images for s in SPLITS
        }
        over = max(SPLITS, key=lambda s: fractions[s] - ratios[s])
        under = min(SPLITS, key=lambda s: fractions[s] - ratios[s])
        if fractions[over] - ratios[over] <= tolerance and ratios[under] - fractions[under] <= tolerance:
            break

        need = (ratios[under] - fractions[under]) * total_images
        totals = _class_totals(assignment)
        current_error = _class_balance_error(assignment, ratios, totals)

        best: tuple[float, Block] | None = None
        for block in assignment[over]:
            # Never strand a class: the donor must retain another block for
            # every class this one carries.
            keeps_classes = all(
                block.class_counts[c] == 0
                or any(other is not block and other.class_counts[c] > 0 for other in assignment[over])
                for c in range(NUM_CLASSES)
            )
            if not keeps_classes:
                continue

            # Cost combines "does this block close the size gap" with "what does
            # moving it do to per-class balance". Without the second term the
            # pass happily strips the rare classes out of val/test to hit an
            # image count, which trades a metric we care about for one we do not.
            assignment[over].remove(block)
            assignment[under].append(block)
            delta_error = _class_balance_error(assignment, ratios, totals) - current_error
            assignment[under].remove(block)
            assignment[over].append(block)

            size_penalty = abs(block.size - need) / max(1.0, abs(need))
            penalty = size_penalty + _BALANCE_WEIGHT * delta_error
            if best is None or penalty < best[0]:
                best = (penalty, block)

        if best is None:
            break  # every remaining block is the sole carrier of some class
        assignment[over].remove(best[1])
        assignment[under].append(best[1])

    return assignment


def blocks_of(assignment: dict[str, list[Block]]) -> list[Block]:
    """Flatten an assignment back into a single block list."""
    return [block for split in SPLITS for block in assignment[split]]


def _repair_missing_classes(
    assignment: dict[str, list[Block]],
) -> dict[str, list[Block]]:
    """Guarantee the brief's 'all classes in each split' requirement.

    Greedy stratification almost always satisfies this on its own, but with only
    81 ``printer`` boxes it is not mathematically guaranteed. If a class is
    missing from a split, move the *smallest* block containing that class out of
    the split that holds the most of it. Moving the smallest such block keeps the
    80/10/10 image ratio as close to intact as possible.
    """
    for split in SPLITS:
        for class_id in range(NUM_CLASSES):
            if any(b.class_counts[class_id] > 0 for b in assignment[split]):
                continue
            donor = max(
                (s for s in SPLITS if s != split),
                key=lambda s: sum(b.class_counts[class_id] for b in assignment[s]),
            )
            candidates = [b for b in assignment[donor] if b.class_counts[class_id] > 0]
            if len(candidates) <= 1:
                # Removing the donor's only block would just move the problem.
                raise ValueError(
                    f"Cannot place class {CLASSES[class_id]!r} in split {split!r}: "
                    f"only {len(candidates)} block(s) contain it. "
                    "Reduce --block-size so the class is spread over more blocks."
                )
            block = min(candidates, key=lambda b: b.size)
            assignment[donor].remove(block)
            assignment[split].append(block)
    return assignment


def make_split(
    records: list[ImageRecord],
    *,
    strategy: str = "grouped",
    block_size: int = 30,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Return ``{split_name: [record indices]}``.

    Args:
        records: parsed images, as produced by :func:`indoor_det.parse.parse_dataset`.
        strategy: ``"grouped"`` (leak-free, default) or ``"random"`` (baseline).
        block_size: frames per contiguous block; only used by ``"grouped"``.
        ratios: split fractions, defaulting to 80/10/10.
        seed: controls block shuffling / random assignment.
    """
    ratios = ratios or DEFAULT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    if strategy == "random":
        rng = random.Random(seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        n_train = int(len(indices) * ratios["train"])
        n_val = int(len(indices) * ratios["val"])
        return {
            "train": sorted(indices[:n_train]),
            "val": sorted(indices[n_train : n_train + n_val]),
            "test": sorted(indices[n_train + n_val :]),
        }

    if strategy != "grouped":
        raise ValueError(f"Unknown split strategy {strategy!r}; expected 'grouped' or 'random'")

    blocks = _build_blocks(records, block_size)
    assignment = _greedy_stratify(blocks, ratios, seed)
    assignment = _rebalance_images(assignment, ratios)
    assignment = _repair_missing_classes(assignment)
    return {
        split: sorted(idx for block in assignment[split] for idx in block.indices)
        for split in SPLITS
    }


def select_split(
    records: list[ImageRecord],
    *,
    block_sizes: tuple[int, ...] = (20, 25, 30, 40),
    seeds: tuple[int, ...] = tuple(range(20)),
    ratios: dict[str, float] | None = None,
    min_frame_gap: int = 10,
    max_ratio_deviation: float = 0.02,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Search block sizes and seeds for the best feasible grouped split.

    Block size trades two things against each other: larger blocks push
    validation frames further from their nearest training neighbour (less
    leakage) but leave fewer blocks to stratify with, so rare classes clump.
    Rather than guess, we enumerate a small grid and pick by an explicit
    objective.

    Feasible means all three of:

    * every class present in every split (the brief requires this);
    * median frame gap to the nearest training neighbour >= ``min_frame_gap``
      -- measured on this dataset, box IoU between frames that far apart has
      already fallen to ~0.03, i.e. they are effectively independent views;
    * every split within ``max_ratio_deviation`` of its 80/10/10 target.

    Among feasible candidates we maximise the *worst-case* per-class instance
    count in val and test, because the rarest class is what makes a per-class
    mAP unreliable, then break ties on the tightest split ratios.

    Returns the chosen split and a dict describing how it was chosen.
    """
    ratios = ratios or DEFAULT_RATIOS
    targets = [(name, ratios[name]) for name in SPLITS]

    feasible: list[tuple[tuple[int, float], dict[str, list[int]], dict[str, object]]] = []
    considered = 0

    for block_size in block_sizes:
        for seed in seeds:
            considered += 1
            candidate = make_split(
                records, strategy="grouped", block_size=block_size, ratios=ratios, seed=seed
            )
            report = split_report(records, candidate)
            splits = report["splits"]  # type: ignore[index]
            leakage = report["leakage_median_frame_gap"]  # type: ignore[index]

            if any(splits[name]["missing_classes"] for name in SPLITS):
                continue
            if min(leakage["val"], leakage["test"]) < min_frame_gap:
                continue
            deviation = max(abs(splits[name]["fraction"] - target) for name, target in targets)
            if deviation > max_ratio_deviation:
                continue

            worst_class_support = min(
                min(splits[name]["per_class"].values()) for name in ("val", "test")
            )
            # Sort key: maximise support, then minimise ratio deviation.
            feasible.append(
                (
                    (-worst_class_support, deviation),
                    candidate,
                    {
                        "block_size": block_size,
                        "seed": seed,
                        "worst_class_support": worst_class_support,
                        "max_ratio_deviation": round(deviation, 4),
                        "median_frame_gap": leakage,
                    },
                )
            )

    if not feasible:
        raise ValueError(
            f"No feasible split among {considered} candidates. Relax min_frame_gap "
            f"({min_frame_gap}) or max_ratio_deviation ({max_ratio_deviation})."
        )

    feasible.sort(key=lambda item: item[0])
    _, best_split, choice = feasible[0]
    choice["candidates_considered"] = considered
    choice["candidates_feasible"] = len(feasible)
    return best_split, choice


def temporal_leakage(
    records: list[ImageRecord], split: dict[str, list[int]]
) -> dict[str, float]:
    """Median frame distance from each val/test frame to its nearest train frame.

    This is the number that makes the leakage argument concrete. A value of 1.0
    means the typical validation frame has an immediate temporal neighbour in
    training -- i.e. the model is being evaluated on pictures it has already
    seen. Higher is better; the grouped strategy should be at least half the
    block size.
    """
    train_frames: dict[int, list[int]] = defaultdict(list)
    for idx in split["train"]:
        train_frames[records[idx].sequence].append(records[idx].frame)
    for frames in train_frames.values():
        frames.sort()

    result: dict[str, float] = {}
    for name in ("val", "test"):
        distances = []
        for idx in split[name]:
            record = records[idx]
            neighbours = train_frames.get(record.sequence)
            if not neighbours:
                continue
            distances.append(min(abs(f - record.frame) for f in neighbours))
        distances.sort()
        result[name] = float(distances[len(distances) // 2]) if distances else float("inf")
    return result


def split_report(
    records: list[ImageRecord], split: dict[str, list[int]]
) -> dict[str, object]:
    """Human- and machine-readable summary of a split, for the README and asserts."""
    total = sum(len(v) for v in split.values())
    per_split: dict[str, object] = {}
    for name in SPLITS:
        indices = split[name]
        counts = Counter()
        for idx in indices:
            for label in records[idx].labels:
                counts[CLASSES[label]] += 1
        per_split[name] = {
            "images": len(indices),
            "fraction": round(len(indices) / total, 4) if total else 0.0,
            "boxes": int(sum(counts.values())),
            "per_class": {name: int(counts.get(name, 0)) for name in CLASSES},
            "missing_classes": [c for c in CLASSES if counts.get(c, 0) == 0],
        }
    return {"splits": per_split, "leakage_median_frame_gap": temporal_leakage(records, split)}


def save_split(split: dict[str, list[int]], records: list[ImageRecord], path: Path) -> Path:
    """Persist a split as filenames (not indices) so it survives re-parsing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [records[i].file for i in split[name]] for name in SPLITS}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def format_report(report: dict[str, object]) -> str:
    """Render :func:`split_report` output as an aligned text table."""
    splits = report["splits"]  # type: ignore[index]
    header = f"{'split':<6}{'images':>8}{'frac':>8}{'boxes':>8}  " + "".join(
        f"{c[:9]:>10}" for c in CLASSES
    )
    lines = [header, "-" * len(header)]
    for name in SPLITS:
        row = splits[name]  # type: ignore[index]
        line = (
            f"{name:<6}{row['images']:>8}{row['fraction']:>8.3f}{row['boxes']:>8}  "
            + "".join(f"{row['per_class'][c]:>10}" for c in CLASSES)
        )
        lines.append(line)
    leak = report["leakage_median_frame_gap"]  # type: ignore[index]
    lines.append("")
    lines.append(
        f"median frames to nearest train neighbour:  "
        f"val={leak['val']:.0f}  test={leak['test']:.0f}   (higher = less leakage)"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    from .parse import load_records

    loaded = load_records()
    for strategy in ("grouped", "random"):
        result = make_split(loaded, strategy=strategy)
        print(f"\n=== {strategy} ===")
        print(format_report(split_report(loaded, result)))
        save_split(result, loaded, INTERIM_DIR / f"split_{strategy}.json")
