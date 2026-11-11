"""Integrity checks for automatically generated M3 labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from .clean import normalize_arxiv_id
from .data import load_joined_paper


def validate_automatic_labels(
    pool_rows: Iterable[Mapping[str, object]],
    label_rows: Iterable[Mapping[str, object]],
    m1_index: dict[str, Path],
    m2_index: dict[str, Path],
) -> dict[str, int]:
    """Validate complete row coverage and publication-scoped ground truth."""

    pool: dict[str, Mapping[str, object]] = {}
    for row in pool_rows:
        arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
        if not arxiv_id:
            raise ValueError("Automatic pool row is missing arxiv_id")
        if arxiv_id in pool:
            raise ValueError(f"Duplicate automatic-pool publication: {arxiv_id}")
        pool[arxiv_id] = row

    labels: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    accepted = no_match = 0
    for row in label_rows:
        arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
        citation_key = str(row.get("citation_key") or "").strip()
        if not arxiv_id or not citation_key:
            raise ValueError("Automatic label requires arxiv_id and citation_key")
        if arxiv_id not in pool:
            raise ValueError(f"Label publication is outside the automatic pool: {arxiv_id}")
        if citation_key in labels[arxiv_id]:
            raise ValueError(f"Duplicate automatic label: {arxiv_id}/{citation_key}")
        if row.get("label_source") != "automatic":
            raise ValueError(f"Wrong label_source for {arxiv_id}/{citation_key}")
        groundtruth = normalize_arxiv_id(row.get("groundtruth_arxiv_id"))
        is_accepted = bool(row.get("accepted"))
        if is_accepted != bool(groundtruth):
            raise ValueError(f"accepted/groundtruth mismatch for {arxiv_id}/{citation_key}")
        accepted += int(is_accepted)
        no_match += int(not is_accepted)
        labels[arxiv_id][citation_key] = row

    if set(labels) != set(pool):
        missing = sorted(set(pool) - set(labels))[:5]
        raise ValueError(f"Automatic publications without label rows: {missing}")

    expected_rows = 0
    for arxiv_id, pool_row in pool.items():
        entries, references = load_joined_paper(arxiv_id, m1_index, m2_index)
        expected_keys = {entry.citation_key for entry in entries}
        actual_keys = set(labels[arxiv_id])
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)[:3]
            extra = sorted(actual_keys - expected_keys)[:3]
            raise ValueError(
                f"Citation coverage mismatch for {arxiv_id}; missing={missing}, extra={extra}"
            )
        declared_count = int(pool_row.get("bibtex_entries") or -1)
        if declared_count != len(expected_keys):
            raise ValueError(
                f"Pool BibTeX count mismatch for {arxiv_id}: "
                f"declared={declared_count}, actual={len(expected_keys)}"
            )
        reference_ids = {reference.ref_arxiv_id for reference in references}
        for citation_key, row in labels[arxiv_id].items():
            groundtruth = normalize_arxiv_id(row.get("groundtruth_arxiv_id"))
            if groundtruth and groundtruth not in reference_ids:
                raise ValueError(
                    f"Ground truth is outside publication candidates: "
                    f"{arxiv_id}/{citation_key} -> {groundtruth}"
                )
        expected_rows += len(expected_keys)

    return {
        "automatic_publications": len(pool),
        "automatic_label_rows": expected_rows,
        "labels_accepted": accepted,
        "labels_no_match": no_match,
    }
