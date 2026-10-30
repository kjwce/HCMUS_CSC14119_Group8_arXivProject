"""Parallel, resumable Milestone 2 runner.

The runner is ready for the teammate-owned modules ``gather.py``, ``normalize.py``
and ``bib.py``. Use ``--inventory`` before those modules arrive to validate an M1
corpus without parsing it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .hierarchy import build_tree
from .models import Node
from .store import PublicationStore, normalize_arxiv_id


def _version_number(value: str | int) -> int:
    return int(str(value).lower().removeprefix("v"))


def discover_papers(input_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_dir() and (path / "metadata.json").is_file()
        ),
        key=lambda path: path.name,
    )


def load_version_specs(paper_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    metadata = json.loads((paper_dir / "metadata.json").read_text(encoding="utf-8"))
    arxiv_id = normalize_arxiv_id(str(metadata.get("arxiv_id") or paper_dir.name))
    specs = metadata.get("versions") or []
    if not specs:
        specs = [
            {"version": child.name, "source_status": "unknown"}
            for child in paper_dir.iterdir()
            if child.is_dir() and child.parent == paper_dir and child.name.lower().startswith("v")
        ]
    return arxiv_id, sorted(specs, key=lambda item: _version_number(item["version"]))


def inventory(input_root: Path) -> dict[str, Any]:
    papers = discover_papers(input_root)
    status_counts: dict[str, int] = {}
    missing_version_dirs: list[str] = []
    versions = 0
    tex_files = 0
    bib_files = 0
    for paper_dir in papers:
        arxiv_id, specs = load_version_specs(paper_dir)
        versions += len(specs)
        for spec in specs:
            status = str(spec.get("source_status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            number = _version_number(spec["version"])
            version_dir = paper_dir / f"v{number}"
            if not version_dir.is_dir():
                missing_version_dirs.append(f"{arxiv_id}/v{number}")
                continue
            tex_files += sum(1 for _ in version_dir.rglob("*.tex"))
            bib_files += sum(1 for _ in version_dir.rglob("*.bib"))
    return {
        "papers": len(papers),
        "versions": versions,
        "version_status_counts": dict(sorted(status_counts.items())),
        "missing_version_directories": missing_version_dirs,
        "tex_files": tex_files,
        "bib_files": bib_files,
    }


def _load_tv2_modules() -> tuple[Any, Any, Any]:
    package = __package__ or "src"
    missing: list[str] = []
    loaded: list[Any] = []
    for name in ("gather", "normalize", "bib"):
        try:
            loaded.append(importlib.import_module(f"{package}.{name}"))
        except ModuleNotFoundError as error:
            if error.name == f"{package}.{name}":
                missing.append(f"src/{name}.py")
            else:
                raise
    if missing:
        raise RuntimeError(
            "Missing teammate modules: " + ", ".join(missing) + ". "
            "Run with --inventory for now or copy the TV2 modules into src/."
        )
    return loaded[0], loaded[1], loaded[2]


def _entry_to_dict(entry: Any, arxiv_id: str) -> dict[str, Any]:
    if is_dataclass(entry):
        row = asdict(entry)
    elif isinstance(entry, dict):
        row = dict(entry)
    else:
        row = dict(vars(entry))
    row.setdefault("arxiv_id", arxiv_id)
    return row


class MemorySampler:
    """Periodically sample RSS for the runner and all worker descendants."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except ImportError:
            self._psutil = None

    def _sample_once(self) -> float:
        if self._psutil is None:
            return 0.0
        process = self._psutil.Process(os.getpid())
        processes = [process]
        try:
            processes.extend(process.children(recursive=True))
        except self._psutil.Error:
            pass
        total = 0
        for child in processes:
            try:
                total += child.memory_info().rss
            except self._psutil.Error:
                continue
        return total / (1024 * 1024)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.samples_mb.append(self._sample_once())

    def start(self) -> None:
        if self._psutil is None:
            return
        self.samples_mb.append(self._sample_once())
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 4))
        self.samples_mb.append(self._sample_once())

    @property
    def peak_mb(self) -> float | None:
        return round(max(self.samples_mb), 3) if self.samples_mb else None

    @property
    def average_mb(self) -> float | None:
        return round(sum(self.samples_mb) / len(self.samples_mb), 3) if self.samples_mb else None


def _normalize_tree(node: Node, normalize_module: Any, rename_map: dict[str, str]) -> None:
    content = node.content
    content = normalize_module.rename_cites(content, rename_map)
    content = normalize_module.normalize_math(content)
    content = normalize_module.clean(content)
    node.content = content
    for child in node.children:
        _normalize_tree(child, normalize_module, rename_map)


def _rename_map_for_version(rename_maps: Any, version: int) -> dict[str, str]:
    if not isinstance(rename_maps, dict):
        return {}
    per_version = rename_maps.get(version) or rename_maps.get(f"v{version}")
    if isinstance(per_version, dict):
        return per_version
    # Backward-compatible global map from the original two-person plan.
    if all(isinstance(value, str) for value in rename_maps.values()):
        return rename_maps
    return {}


def process_paper(
    paper_dir_text: str,
    output_root_text: str,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    paper_dir = Path(paper_dir_text)
    output_root = Path(output_root_text)
    arxiv_id, specs = load_version_specs(paper_dir)
    publication_dir = output_root / arxiv_id
    if (publication_dir / "_SUCCESS").is_file() and not force:
        return {"arxiv_id": arxiv_id, "status": "skipped", "seconds": 0.0}

    gather_module, normalize_module, bib_module = _load_tv2_modules()
    gathered: dict[int, Any] = {}
    entries_by_version: dict[int, Any] = {}
    version_status: dict[int, str] = {}

    for spec in specs:
        version = _version_number(spec["version"])
        status = str(spec.get("source_status") or "unknown")
        version_dir = paper_dir / f"v{version}"
        if status != "ok" or not version_dir.is_dir():
            version_status[version] = "empty" if status != "ok" else "missing_directory"
            continue
        result = gather_module.flatten(version_dir)
        if not getattr(result, "text", "").strip():
            version_status[version] = "empty_gathered_text"
            continue
        gathered[version] = result
        entries_by_version[version] = bib_module.extract(version_dir, result.text)
        version_status[version] = "gathered"

    unified_entries, rename_maps = bib_module.dedup(entries_by_version)
    store = PublicationStore(arxiv_id)
    gather_stats: dict[str, Any] = {}

    for spec in specs:
        version = _version_number(spec["version"])
        if version not in gathered:
            store.record_empty_version(version)
            continue
        result = gathered[version]
        tree = build_tree(result.text, normalize_module.split_sentences)
        _normalize_tree(tree, normalize_module, _rename_map_for_version(rename_maps, version))
        store.add_version(version, tree)
        gather_stats[f"v{version}"] = {
            "used_files": [str(path) for path in result.used_files],
            "unused_files": [str(path) for path in result.unused_files],
            "used_file_count": len(result.used_files),
            "unused_file_count": len(result.unused_files),
        }
        version_status[version] = "parsed"

    if isinstance(unified_entries, dict):
        unified_entries = list(unified_entries.values())
    bibtex_rows = [_entry_to_dict(entry, arxiv_id) for entry in unified_entries]
    entries_before_dedup = sum(len(entries) for entries in entries_by_version.values())
    store.write(
        output_root,
        bibtex_rows,
        {
            "version_status": version_status,
            "gather": gather_stats,
            "bibtex_entries_before_dedup": entries_before_dedup,
            "bibtex_entries_after_dedup": len(bibtex_rows),
        },
    )
    return {
        "arxiv_id": arxiv_id,
        "status": "ok",
        "seconds": round(time.perf_counter() - started, 6),
        "versions": len(specs),
        "versions_parsed": store.stats.versions_parsed,
        "empty_versions": len(store.stats.empty_versions),
        "elements_before_dedup": store.stats.elements_before_dedup,
        "elements_after_dedup": len(store.elements),
        "bibtex_entries_before_dedup": entries_before_dedup,
        "bibtex_entries_after_dedup": len(bibtex_rows),
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run(args: argparse.Namespace) -> int:
    input_root = Path(args.input).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    if args.inventory:
        result = inventory(input_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    # Fail before starting workers if the TV2 contract is not available.
    _load_tv2_modules()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    papers = discover_papers(input_root)
    if args.limit is not None:
        papers = papers[: args.limit]

    started = time.perf_counter()
    memory = MemorySampler()
    memory.start()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_paper, str(paper), str(output_root), args.force): paper
                for paper in papers
            }
            for index, future in enumerate(as_completed(futures), start=1):
                paper = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{index}/{len(futures)}] {result['arxiv_id']}: {result['status']}")
                except Exception as error:  # keep the full corpus run alive
                    record = {
                        "paper": paper.name,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(record)
                    print(f"[{index}/{len(futures)}] {paper.name}: ERROR {error}", file=sys.stderr)
    finally:
        memory.stop()

    wall_time = time.perf_counter() - started
    successful = [row for row in results if row["status"] == "ok"]
    summary = {
        "papers_requested": len(papers),
        "papers_processed": len(successful),
        "papers_skipped": sum(row["status"] == "skipped" for row in results),
        "papers_failed": len(errors),
        "parse_success_rate": len(successful) / len(papers) if papers else 0.0,
        "wall_time_s": round(wall_time, 3),
        "avg_seconds_per_processed_paper": (
            round(sum(row["seconds"] for row in successful) / len(successful), 6)
            if successful
            else None
        ),
        "versions_parsed": sum(row.get("versions_parsed", 0) for row in successful),
        "empty_versions": sum(row.get("empty_versions", 0) for row in successful),
        "elements_before_dedup": sum(
            row.get("elements_before_dedup", 0) for row in successful
        ),
        "elements_after_dedup": sum(
            row.get("elements_after_dedup", 0) for row in successful
        ),
        "bibtex_entries_before_dedup": sum(
            row.get("bibtex_entries_before_dedup", 0) for row in successful
        ),
        "bibtex_entries_after_dedup": sum(
            row.get("bibtex_entries_after_dedup", 0) for row in successful
        ),
        "peak_ram_mb": memory.peak_mb,
        "avg_ram_mb": memory.average_mb,
        "output_bytes": _directory_size(output_root),
        "workers": workers,
    }
    (output_root / "m2_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="M1 raw directory")
    parser.add_argument("--output", default="processed", help="M2 output directory")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process the first N papers")
    parser.add_argument("--force", action="store_true", help="Reprocess completed papers")
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Inspect M1 data without requiring the teammate modules",
    )
    return parser


def main() -> int:
    return run(build_argument_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

