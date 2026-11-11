"""Quality audit and final recommendations for manual-label candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .clean import normalize_text
from .data import load_joined_paper


def audit_manual_shortlist(
    shortlist: Iterable[dict[str, object]],
    m1_index: dict[str, Path],
    m2_index: dict[str, Path],
    *,
    final_per_range: int = 5,
) -> list[dict[str, object]]:
    """Add observable quality measures and five final choices per range.

    Four easy publications plus one medium publication are preferred when
    available.  This keeps the manual workload reasonable without making the
    entire evaluation pool consist only of exact-title matches.
    """

    audited: list[dict[str, object]] = []
    by_range: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source_row in shortlist:
        row = dict(source_row)
        arxiv_id = str(row["arxiv_id"])
        entries, references = load_joined_paper(arxiv_id, m1_index, m2_index)
        reference_titles = {normalize_text(reference.title) for reference in references}
        reference_titles.discard("")
        exact_matches = sum(
            bool(normalize_text(entry.title))
            and normalize_text(entry.title) in reference_titles
            for entry in entries
        )
        complete_entries = sum(
            bool(entry.title and entry.authors and entry.year is not None) for entry in entries
        )
        completeness = complete_entries / max(1, len(entries))
        exact_coverage = exact_matches / max(1, len(entries))
        if exact_matches >= 20 and completeness >= 0.90:
            difficulty = "easy"
        elif exact_matches >= 10:
            difficulty = "medium"
        else:
            difficulty = "hard"
        row.update(
            {
                "original_recommended": bool(row.pop("recommended", False)),
                "exact_title_matches": exact_matches,
                "exact_title_coverage_pct": round(100 * exact_coverage, 2),
                "complete_bibtex_entries": complete_entries,
                "bibtex_completeness_pct": round(100 * completeness, 2),
                "difficulty": difficulty,
                "final_recommended": False,
                "final_rank": None,
            }
        )
        audited.append(row)
        by_range[str(row["range"])].append(row)

    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}

    def quality_key(row: dict[str, object]) -> tuple[float, int, str]:
        return (
            -float(row["exact_title_coverage_pct"]),
            int(row["bibtex_entries"]),
            str(row["arxiv_id"]),
        )

    for range_name, rows in by_range.items():
        easy = sorted((row for row in rows if row["difficulty"] == "easy"), key=quality_key)
        medium = sorted(
            (row for row in rows if row["difficulty"] == "medium"), key=quality_key
        )
        chosen = easy[: max(0, final_per_range - 1)]
        if medium and len(chosen) < final_per_range:
            chosen.append(medium[0])
        remaining = sorted(
            (row for row in rows if row not in chosen),
            key=lambda row: (difficulty_rank[str(row["difficulty"])], *quality_key(row)),
        )
        chosen.extend(remaining[: final_per_range - len(chosen)])
        if len(chosen) != final_per_range:
            raise ValueError(
                f"Range {range_name} has only {len(chosen)} auditable publications; "
                f"{final_per_range} required"
            )
        for rank, row in enumerate(chosen, 1):
            row["final_recommended"] = True
            row["final_rank"] = rank

    return sorted(
        audited,
        key=lambda row: (
            str(row["range"]),
            not bool(row["final_recommended"]),
            int(row["final_rank"] or 999),
            int(row["shortlist_rank"]),
        ),
    )
