"""Assign stable element IDs and persist per-publication Milestone 2 tables."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import Node


@dataclass(frozen=True, slots=True)
class ElementRecord:
    element_id: str
    arxiv_id: str
    kind: str
    content: str


@dataclass(frozen=True, slots=True)
class HierarchyRecord:
    arxiv_id: str
    version: int
    element_id: str
    parent_id: str


@dataclass(slots=True)
class PublicationStats:
    arxiv_id: str
    versions_seen: int = 0
    versions_parsed: int = 0
    empty_versions: list[int] = field(default_factory=list)
    elements_before_dedup: int = 0
    elements_after_dedup: int = 0
    hierarchy_edges: int = 0
    max_depth: int = 0
    depth_counts: Counter[int] = field(default_factory=Counter)
    kind_counts_before_dedup: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["depth_counts"] = dict(sorted(self.depth_counts.items()))
        result["kind_counts_before_dedup"] = dict(sorted(self.kind_counts_before_dedup.items()))
        return result


def normalize_arxiv_id(arxiv_id: str) -> str:
    return arxiv_id.strip().replace(".", "-")


def _signature(kind: str, content: str, ordinal: int) -> str:
    payload = f"{kind}\0{content}\0{ordinal}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:20]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl_atomic(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


class PublicationStore:
    """Shared element store for all versions of one publication."""

    def __init__(self, arxiv_id: str) -> None:
        self.arxiv_id = normalize_arxiv_id(arxiv_id)
        self.elements: dict[str, ElementRecord] = {}
        self.hierarchy: set[HierarchyRecord] = set()
        self.stats = PublicationStats(arxiv_id=self.arxiv_id)

    def record_empty_version(self, version: int) -> None:
        self.stats.versions_seen += 1
        if version not in self.stats.empty_versions:
            self.stats.empty_versions.append(version)

    def add_version(self, version: int, root: Node) -> None:
        """Add one cleaned tree and deduplicate exact content across versions.

        An occurrence ordinal is included in the discriminant. Consequently, the
        first identical sentence in v1 shares an ID with the first identical
        sentence in v2, while two repeated sentences in the same version remain
        distinguishable.
        """

        self.stats.versions_seen += 1
        self.stats.versions_parsed += 1
        occurrence_counts: Counter[tuple[str, str]] = Counter()
        node_ids: dict[int, str] = {}

        for node, depth in root.walk():
            self.stats.elements_before_dedup += 1
            self.stats.kind_counts_before_dedup[node.kind] += 1
            self.stats.depth_counts[depth] += 1
            self.stats.max_depth = max(self.stats.max_depth, depth)

            if node is root:
                element_id = f"{self.arxiv_id}#root-v{version}"
                content = f"Version {version} root"
            else:
                content = node.content.strip()
                key = (node.kind, content)
                ordinal = occurrence_counts[key]
                occurrence_counts[key] += 1
                element_id = f"{self.arxiv_id}#{node.kind}-{_signature(node.kind, content, ordinal)}"

            existing = self.elements.get(element_id)
            if existing and (existing.kind != node.kind or existing.content != content):
                raise RuntimeError(f"Element hash collision for {element_id}")
            self.elements.setdefault(
                element_id,
                ElementRecord(element_id, self.arxiv_id, node.kind, content),
            )
            node_ids[id(node)] = element_id

        def add_edges(parent: Node) -> None:
            parent_id = node_ids[id(parent)]
            for child in parent.children:
                child_id = node_ids[id(child)]
                self.hierarchy.add(
                    HierarchyRecord(self.arxiv_id, version, child_id, parent_id)
                )
                add_edges(child)

        add_edges(root)
        self.stats.elements_after_dedup = len(self.elements)
        self.stats.hierarchy_edges = len(self.hierarchy)

    def write(
        self,
        output_root: str | Path,
        bibtex_rows: Iterable[Any] = (),
        extra_stats: dict[str, Any] | None = None,
    ) -> Path:
        publication_dir = Path(output_root) / self.arxiv_id
        publication_dir.mkdir(parents=True, exist_ok=True)

        elements = sorted(self.elements.values(), key=lambda row: row.element_id)
        hierarchy = sorted(
            self.hierarchy,
            key=lambda row: (row.version, row.parent_id, row.element_id),
        )
        _write_jsonl_atomic(publication_dir / "elements.jsonl", elements)
        _write_jsonl_atomic(publication_dir / "hierarchy.jsonl", hierarchy)
        _write_jsonl_atomic(publication_dir / "bibtex.jsonl", bibtex_rows)

        stats = self.stats.to_dict()
        if extra_stats:
            stats.update(_jsonable(extra_stats))
        temporary = publication_dir / "stats.json.tmp"
        temporary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, publication_dir / "stats.json")
        (publication_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        return publication_dir

