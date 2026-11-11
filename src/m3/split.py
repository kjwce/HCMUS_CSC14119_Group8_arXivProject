"""Publication-level train/valid/test splitting without information leakage."""

from __future__ import annotations

import random


def split_publications(
    manual_ids: set[str],
    automatic_ids: set[str],
    *,
    holdout_per_source: int = 5,
    seed: int = 14119,
) -> list[dict[str, str]]:
    """Return a deterministic split that follows the five-member requirement."""

    overlap = manual_ids & automatic_ids
    if overlap:
        sample = ", ".join(sorted(overlap)[:3])
        raise ValueError(f"Manual and automatic pools overlap: {sample}")
    # The statement requires five manually labelled publications per member.
    # ``holdout_per_source`` equals the group size because validation and test
    # each receive one manual publication per member.
    required_manual = 5 * holdout_per_source
    required_automatic = 2 * holdout_per_source
    if len(manual_ids) < required_manual:
        raise ValueError(
            f"Need at least {required_manual} manual publications, got {len(manual_ids)}"
        )
    if len(automatic_ids) < required_automatic:
        raise ValueError(
            f"Need at least {required_automatic} automatic publications, got {len(automatic_ids)}"
        )

    rng = random.Random(seed)
    manual = sorted(manual_ids)
    automatic = sorted(automatic_ids)
    rng.shuffle(manual)
    rng.shuffle(automatic)

    rows: list[dict[str, str]] = []
    for label_source, values in (("manual", manual), ("automatic", automatic)):
        test = set(values[:holdout_per_source])
        valid = set(values[holdout_per_source : 2 * holdout_per_source])
        for arxiv_id in values:
            partition = "test" if arxiv_id in test else "valid" if arxiv_id in valid else "train"
            rows.append(
                {
                    "arxiv_id": arxiv_id,
                    "label_source": label_source,
                    "partition": partition,
                }
            )
    return sorted(rows, key=lambda row: row["arxiv_id"])
