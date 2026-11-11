"""Typed records shared by the Milestone 3 preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BibEntry:
    """One unified BibTeX row produced by Milestone 2."""

    arxiv_id: str
    citation_key: str
    entry_type: str | None
    title: str | None
    authors: tuple[str, ...]
    year: int | None
    doi: str | None
    journal: str | None
    url: str | None


@dataclass(frozen=True, slots=True)
class Reference:
    """One arXiv-keyed Semantic Scholar reference produced by Milestone 1."""

    arxiv_id: str
    ref_arxiv_id: str
    title: str | None
    authors: tuple[str, ...]
    year: int | None
    submission_date: str | None
    semantic_scholar_id: str | None
    doi: str | None = None


@dataclass(frozen=True, slots=True)
class InventoryRow:
    """Counts used to choose publications for manual and automatic labelling."""

    arxiv_id: str
    bibtex_entries: int
    arxiv_references: int

    @property
    def eligible_for_manual_labelling(self) -> bool:
        # The statement requires at least 20 labelled entries per manually
        # labelled publication.  Requiring at least one candidate reference
        # avoids selecting a publication whose matrix is entirely empty.
        return self.bibtex_entries >= 20 and self.arxiv_references > 0

    @property
    def eligible_for_automatic_labelling(self) -> bool:
        return self.bibtex_entries > 0 and self.arxiv_references > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "arxiv_id": self.arxiv_id,
            "bibtex_entries": self.bibtex_entries,
            "arxiv_references": self.arxiv_references,
            "eligible_for_manual_labelling": self.eligible_for_manual_labelling,
            "eligible_for_automatic_labelling": self.eligible_for_automatic_labelling,
        }
