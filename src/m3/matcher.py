"""Auditable classic heuristic matcher used for automatic M3 labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from .clean import author_surnames, extract_arxiv_id, normalize_text, title_tokens
from .models import BibEntry, Reference


@dataclass(frozen=True, slots=True)
class CandidateScore:
    ref_arxiv_id: str
    score: float
    arxiv_id_exact: float
    title_sequence: float
    title_jaccard: float
    author_jaccard: float
    year_score: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PreparedEntry:
    source: BibEntry
    embedded_arxiv_id: str | None
    title: str
    title_tokens: frozenset[str]
    author_surnames: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PreparedReference:
    source: Reference
    title: str
    title_tokens: frozenset[str]
    author_surnames: frozenset[str]


def _prepare_entry(entry: BibEntry) -> _PreparedEntry:
    return _PreparedEntry(
        source=entry,
        embedded_arxiv_id=extract_arxiv_id(entry.url, entry.title),
        title=normalize_text(entry.title),
        title_tokens=title_tokens(entry.title),
        author_surnames=author_surnames(entry.authors),
    )


def _prepare_reference(reference: Reference) -> _PreparedReference:
    return _PreparedReference(
        source=reference,
        title=normalize_text(reference.title),
        title_tokens=title_tokens(reference.title),
        author_surnames=author_surnames(reference.authors),
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _year_score(left: int | None, right: int | None) -> float:
    if left is None or right is None:
        return 0.0
    difference = abs(left - right)
    if difference == 0:
        return 1.0
    if difference == 1:
        return 0.5
    return 0.0


def _score_prepared(
    entry: _PreparedEntry,
    reference: _PreparedReference,
) -> CandidateScore:
    id_exact = float(
        bool(
            entry.embedded_arxiv_id
            and entry.embedded_arxiv_id == reference.source.ref_arxiv_id
        )
    )
    has_shared_title_token = bool(entry.title_tokens & reference.title_tokens)
    title_sequence = (
        SequenceMatcher(None, entry.title, reference.title, autojunk=False).ratio()
        if entry.title and reference.title and has_shared_title_token
        else 0.0
    )
    title_jaccard = _jaccard(entry.title_tokens, reference.title_tokens)
    author_jaccard = _jaccard(entry.author_surnames, reference.author_surnames)
    year = _year_score(entry.source.year, reference.source.year)

    if id_exact:
        total = 1.0
    else:
        # Title is the most discriminative field.  Author and year provide
        # independent evidence without making missing metadata fatal.
        total = (
            0.40 * title_sequence
            + 0.30 * title_jaccard
            + 0.20 * author_jaccard
            + 0.10 * year
        )
    return CandidateScore(
        ref_arxiv_id=reference.source.ref_arxiv_id,
        score=round(total, 6),
        arxiv_id_exact=id_exact,
        title_sequence=round(title_sequence, 6),
        title_jaccard=round(title_jaccard, 6),
        author_jaccard=round(author_jaccard, 6),
        year_score=year,
    )


def score_candidate(entry: BibEntry, reference: Reference) -> CandidateScore:
    return _score_prepared(_prepare_entry(entry), _prepare_reference(reference))


def rank_candidates(entry: BibEntry, references: list[Reference]) -> list[CandidateScore]:
    prepared_entry = _prepare_entry(entry)
    ranked = [
        _score_prepared(prepared_entry, _prepare_reference(reference))
        for reference in references
    ]
    return sorted(ranked, key=lambda row: (-row.score, row.ref_arxiv_id))


def _automatic_label_prepared(
    entry: _PreparedEntry,
    references: list[_PreparedReference],
    *,
    threshold: float,
    minimum_margin: float,
) -> dict[str, object]:
    # An explicit arXiv identifier is definitive and avoids scoring every
    # candidate for the common easy case.
    if entry.embedded_arxiv_id:
        direct = next(
            (
                reference
                for reference in references
                if reference.source.ref_arxiv_id == entry.embedded_arxiv_id
            ),
            None,
        )
        if direct:
            best = CandidateScore(
                ref_arxiv_id=direct.source.ref_arxiv_id,
                score=1.0,
                arxiv_id_exact=1.0,
                title_sequence=0.0,
                title_jaccard=0.0,
                author_jaccard=0.0,
                year_score=0.0,
            )
            return {
                "arxiv_id": entry.source.arxiv_id,
                "citation_key": entry.source.citation_key,
                "groundtruth_arxiv_id": direct.source.ref_arxiv_id,
                "label_source": "automatic",
                "label_method": "embedded_arxiv_id",
                "confidence": 1.0,
                "runner_up_score": 0.0,
                "margin": 1.0,
                "candidate_count": len(references),
                "accepted": True,
                "best_candidate_features": best.as_dict(),
            }

    ranked = sorted(
        (_score_prepared(entry, reference) for reference in references),
        key=lambda row: (-row.score, row.ref_arxiv_id),
    )
    best = ranked[0] if ranked else None
    runner_up_score = ranked[1].score if len(ranked) > 1 else 0.0
    margin = (best.score - runner_up_score) if best else 0.0
    accepted = bool(best and best.score >= threshold and margin >= minimum_margin)
    return {
        "arxiv_id": entry.source.arxiv_id,
        "citation_key": entry.source.citation_key,
        "groundtruth_arxiv_id": best.ref_arxiv_id if accepted else None,
        "label_source": "automatic",
        "label_method": "weighted_string_similarity",
        "confidence": best.score if best else 0.0,
        "runner_up_score": runner_up_score,
        "margin": round(margin, 6),
        "candidate_count": len(ranked),
        "accepted": accepted,
        "best_candidate_features": best.as_dict() if best else None,
    }


def automatic_label(
    entry: BibEntry,
    references: list[Reference],
    *,
    threshold: float = 0.68,
    minimum_margin: float = 0.05,
) -> dict[str, object]:
    """Label one entry with one reference or an explicit no-match value."""

    return _automatic_label_prepared(
        _prepare_entry(entry),
        [_prepare_reference(reference) for reference in references],
        threshold=threshold,
        minimum_margin=minimum_margin,
    )


def label_publication(
    entries: list[BibEntry],
    references: list[Reference],
    *,
    threshold: float = 0.68,
    minimum_margin: float = 0.05,
) -> list[dict[str, object]]:
    """Label every BibTeX entry, including explicit no-match rows."""

    prepared_references = [_prepare_reference(reference) for reference in references]
    return [
        _automatic_label_prepared(
            _prepare_entry(entry),
            prepared_references,
            threshold=threshold,
            minimum_margin=minimum_margin,
        )
        for entry in entries
    ]
