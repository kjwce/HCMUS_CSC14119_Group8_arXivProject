"""Deterministic, range-balanced publication selection for manual labels."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping


def select_manual_shortlist(
    inventory: Mapping[str, Mapping[str, object]],
    manifests: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    per_range: int = 10,
    seed: int = 14119,
    min_bibtex: int = 20,
    max_bibtex: int = 60,
    min_references: int = 8,
    max_references: int = 40,
) -> list[dict[str, object]]:
    """Choose manageable publications while preserving each member's range.

    The shortlist is intentionally larger than the required five publications
    per member so a failed manual audit can be replaced without rerunning the
    corpus inventory.
    """

    if per_range < 1:
        raise ValueError("per_range must be positive")
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for range_name, manifest_rows in sorted(manifests.items()):
        candidates: list[dict[str, object]] = []
        for metadata in manifest_rows:
            arxiv_id = str(metadata.get("arxiv_id") or "")
            counts = inventory.get(arxiv_id)
            if not counts or not counts.get("eligible_for_manual_labelling"):
                continue
            bibtex_entries = int(counts["bibtex_entries"])
            references = int(counts["arxiv_references"])
            if not (
                min_bibtex <= bibtex_entries <= max_bibtex
                and min_references <= references <= max_references
            ):
                continue
            candidates.append(
                {
                    "range": range_name,
                    "arxiv_id": arxiv_id,
                    "bibtex_entries": bibtex_entries,
                    "arxiv_references": references,
                    "title": metadata.get("title"),
                    "authors": metadata.get("authors") or [],
                    "primary_category": metadata.get("primary_category"),
                }
            )
        # A separate seeded shuffle per range avoids selecting only one date or
        # category while remaining exactly reproducible.
        rng = random.Random(f"{seed}:{range_name}")
        rng.shuffle(candidates)
        if len(candidates) < per_range:
            raise ValueError(
                f"Range {range_name} has only {len(candidates)} candidates; "
                f"{per_range} requested"
            )
        for rank, row in enumerate(candidates[:per_range], 1):
            arxiv_id = str(row["arxiv_id"])
            if arxiv_id in seen:
                raise ValueError(f"Publication occurs in more than one range: {arxiv_id}")
            seen.add(arxiv_id)
            row["shortlist_rank"] = rank
            row["recommended"] = rank <= 5
            selected.append(row)
    return selected


def finalize_manual_pool(
    audited_rows: Iterable[Mapping[str, object]],
    *,
    per_range: int = 5,
) -> list[dict[str, object]]:
    """Validate and retain the audited final recommendations."""

    chosen = [dict(row) for row in audited_rows if row.get("final_recommended")]
    by_range: dict[str, list[dict[str, object]]] = {}
    seen: set[str] = set()
    for row in chosen:
        range_name = str(row.get("range") or "")
        arxiv_id = str(row.get("arxiv_id") or "")
        if not range_name or not arxiv_id:
            raise ValueError("Final manual rows require range and arxiv_id")
        if arxiv_id in seen:
            raise ValueError(f"Duplicate manual publication: {arxiv_id}")
        seen.add(arxiv_id)
        by_range.setdefault(range_name, []).append(row)
    if not by_range:
        raise ValueError("No final manual publications were selected")
    wrong = {name: len(rows) for name, rows in by_range.items() if len(rows) != per_range}
    if wrong:
        details = ", ".join(f"{name}={count}" for name, count in sorted(wrong.items()))
        raise ValueError(f"Each range must contain {per_range} publications: {details}")
    result: list[dict[str, object]] = []
    for range_name, rows in sorted(by_range.items()):
        for row in sorted(rows, key=lambda value: int(value.get("final_rank") or 999)):
            result.append(
                {
                    "range": range_name,
                    "arxiv_id": row["arxiv_id"],
                    "bibtex_entries": row["bibtex_entries"],
                    "arxiv_references": row["arxiv_references"],
                    "difficulty": row["difficulty"],
                    "title": row.get("title"),
                }
            )
    return result
