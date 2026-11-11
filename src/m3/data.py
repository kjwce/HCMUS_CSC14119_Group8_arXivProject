"""Readers that join the M1 reference pool with the M2 BibTeX store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .clean import normalize_arxiv_id, normalize_doi
from .models import BibEntry, InventoryRow, Reference


class CorpusDataError(ValueError):
    """Raised when a corpus row violates the documented M1/M2 contract."""


def _paper_root(path: Path) -> Path:
    raw = path / "raw"
    return raw if raw.is_dir() else path


def _index_paper_directories(root: Path, required_file: str) -> dict[str, Path]:
    root = _paper_root(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    indexed: dict[str, Path] = {}
    for folder in root.iterdir():
        if not folder.is_dir() or not (folder / required_file).is_file():
            continue
        arxiv_id = normalize_arxiv_id(folder.name)
        if not arxiv_id:
            continue
        if arxiv_id in indexed:
            raise CorpusDataError(f"Duplicate publication directory for {arxiv_id}")
        indexed[arxiv_id] = folder
    return indexed


def index_corpora(m1_root: Path, m2_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    return (
        _index_paper_directories(m1_root, "references.json"),
        _index_paper_directories(m2_root, "bibtex.jsonl"),
    )


def read_m2_schema_version(m2_root: Path) -> int | None:
    """Read the corpus-level parser schema without opening the large tables."""

    path = m2_root / "m2_stats.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    if not isinstance(row, dict):
        raise CorpusDataError(f"Expected an object in {path}")
    version = row.get("parser_schema_version")
    return _optional_int(version)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CorpusDataError(f"Expected an integer year, got {value!r}") from exc


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value if item not in (None, ""))
    raise CorpusDataError(f"Expected a list of strings, got {type(value).__name__}")


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusDataError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise CorpusDataError(f"Expected an object at {path}:{line_number}")
            yield row


def load_bibtex(paper_dir: Path, arxiv_id: str | None = None) -> list[BibEntry]:
    path = paper_dir / "bibtex.jsonl"
    paper_id = arxiv_id or normalize_arxiv_id(paper_dir.name)
    if not paper_id:
        raise CorpusDataError(f"Cannot determine arXiv ID for {paper_dir}")
    entries: list[BibEntry] = []
    seen_keys: set[str] = set()
    for row in _iter_jsonl(path):
        citation_key = str(row.get("citation_key") or "").strip()
        if not citation_key:
            raise CorpusDataError(f"Missing citation_key in {path}")
        if citation_key in seen_keys:
            raise CorpusDataError(f"Duplicate citation_key {citation_key!r} in {path}")
        seen_keys.add(citation_key)
        row_paper_id = normalize_arxiv_id(row.get("arxiv_id"))
        if row_paper_id and row_paper_id != paper_id:
            raise CorpusDataError(
                f"BibTeX row {citation_key!r} belongs to {row_paper_id}, expected {paper_id}"
            )
        entries.append(
            BibEntry(
                arxiv_id=paper_id,
                citation_key=citation_key,
                entry_type=str(row.get("entry_type")) if row.get("entry_type") else None,
                title=str(row.get("title")) if row.get("title") else None,
                authors=_string_tuple(row.get("authors")),
                year=_optional_int(row.get("year")),
                doi=normalize_doi(row.get("doi")),
                journal=str(row.get("journal")) if row.get("journal") else None,
                url=str(row.get("url")) if row.get("url") else None,
            )
        )
    return entries


def load_references(paper_dir: Path, arxiv_id: str | None = None) -> list[Reference]:
    path = paper_dir / "references.json"
    paper_id = arxiv_id or normalize_arxiv_id(paper_dir.name)
    if not paper_id:
        raise CorpusDataError(f"Cannot determine arXiv ID for {paper_dir}")
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise CorpusDataError(f"Expected a JSON array in {path}")
    references: list[Reference] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise CorpusDataError(f"Expected reference objects in {path}")
        ref_id = normalize_arxiv_id(row.get("arxiv_id") or row.get("ref_arxiv_id"))
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        references.append(
            Reference(
                arxiv_id=paper_id,
                ref_arxiv_id=ref_id,
                title=str(row.get("title") or row.get("ref_title") or "") or None,
                authors=_string_tuple(row.get("authors") or row.get("ref_authors")),
                year=_optional_int(row.get("year")),
                submission_date=(
                    str(row.get("submission_date") or row.get("ref_submission_date") or "") or None
                ),
                semantic_scholar_id=(
                    str(row.get("semantic_scholar_id") or row.get("ref_s2_id") or "") or None
                ),
                doi=normalize_doi(row.get("doi")),
            )
        )
    return references


def inventory_rows(m1_root: Path, m2_root: Path) -> Iterable[InventoryRow]:
    m1_index, m2_index = index_corpora(m1_root, m2_root)
    for arxiv_id in sorted(m1_index.keys() & m2_index.keys()):
        yield InventoryRow(
            arxiv_id=arxiv_id,
            bibtex_entries=len(load_bibtex(m2_index[arxiv_id], arxiv_id)),
            arxiv_references=len(load_references(m1_index[arxiv_id], arxiv_id)),
        )


def load_joined_paper(
    arxiv_id: str,
    m1_index: dict[str, Path],
    m2_index: dict[str, Path],
) -> tuple[list[BibEntry], list[Reference]]:
    canonical = normalize_arxiv_id(arxiv_id)
    if not canonical or canonical not in m1_index or canonical not in m2_index:
        raise KeyError(f"Publication is not present in both M1 and M2: {arxiv_id}")
    return (
        load_bibtex(m2_index[canonical], canonical),
        load_references(m1_index[canonical], canonical),
    )
