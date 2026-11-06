"""Parallel, resumable end-to-end Milestone 2 runner."""

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

PARSER_SCHEMA_VERSION = 3


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
    # A sentence holding only removed markup (``\label{..}``, ``\vspace{..}``)
    # is empty after cleanup and carries no content.
    node.children = [
        child for child in node.children if child.kind != "sentence" or child.content.strip()
    ]


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
        previous: dict[str, Any] | None = None
        try:
            previous = json.loads((publication_dir / "stats.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A marker without readable statistics is incomplete; rebuild it.
            pass
        else:
            if not isinstance(previous, dict):
                previous = None
            elif previous.get("parser_schema_version") != PARSER_SCHEMA_VERSION:
                # Parser fixes can change element IDs and hierarchy.  Outputs
                # from an older schema must not be silently reused.
                previous = None
        if previous is not None:
            unresolved = previous.get("unresolved_citation_keys") or {}
            return {
                "arxiv_id": arxiv_id,
                "status": "skipped",
                "seconds": 0.0,
                "versions": int(previous.get("versions_seen", len(specs))),
                "versions_parsed": int(previous.get("versions_parsed", 0)),
                "empty_versions": len(previous.get("empty_versions") or []),
                "elements_before_dedup": int(previous.get("elements_before_dedup", 0)),
                "elements_after_dedup": int(previous.get("elements_after_dedup", 0)),
                "bibtex_entries_before_dedup": int(
                    previous.get("bibtex_entries_before_dedup", 0)
                ),
                "bibtex_entries_after_dedup": int(
                    previous.get("bibtex_entries_after_dedup", 0)
                ),
                "unresolved_citation_keys": (
                    sum(len(keys) for keys in unresolved.values())
                    if isinstance(unresolved, dict)
                    else int(unresolved)
                ),
            }

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
    if isinstance(unified_entries, dict):
        unified_entries = list(unified_entries.values())
    bibtex_rows = [_entry_to_dict(entry, arxiv_id) for entry in unified_entries]
    unified_keys = {
        str(row.get("citation_key"))
        for row in bibtex_rows
        if row.get("citation_key") is not None
    }
    cite_keys = getattr(normalize_module, "cite_keys", None)
    unresolved_citation_keys: dict[str, list[str]] = {}
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
        if callable(cite_keys):
            cited = {
                key
                for node, _ in tree.walk()
                for key in cite_keys(node.content)
            }
            missing = sorted(key for key in cited if key != "*" and key not in unified_keys)
            if missing:
                unresolved_citation_keys[f"v{version}"] = missing
        store.add_version(version, tree)
        gather_stats[f"v{version}"] = {
            "root": result.root,
            "used_files": [str(path) for path in result.used_files],
            "unused_files": [str(path) for path in result.unused_files],
            "used_file_count": len(result.used_files),
            "unused_file_count": len(result.unused_files),
            "missing_files": list(getattr(result, "missing_files", [])),
            "cyclic_includes": list(getattr(result, "cyclic_includes", [])),
            "non_utf8_files": list(getattr(result, "non_utf8_files", [])),
            "root_candidates": list(getattr(result, "root_candidates", [])),
        }
        version_status[version] = "parsed"

    entries_before_dedup = sum(len(entries) for entries in entries_by_version.values())
    source_counter = getattr(bib_module, "source_counts", None)
    bibtex_source_counts = source_counter(entries_by_version) if callable(source_counter) else {}

    store.write(
        output_root,
        bibtex_rows,
        {
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "version_status": version_status,
            "gather": gather_stats,
            "bibtex_entries_before_dedup": entries_before_dedup,
            "bibtex_entries_after_dedup": len(bibtex_rows),
            "bibtex_source_counts_before_dedup": bibtex_source_counts,
            "unresolved_citation_keys": unresolved_citation_keys,
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
        "unresolved_citation_keys": sum(
            len(keys) for keys in unresolved_citation_keys.values()
        ),
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_paper_stats_jsonl(output_root: Path, arxiv_ids: list[str]) -> int:
    """Collect every completed publication's full ``stats.json`` into one file."""

    destination = output_root / "paper_stats.jsonl"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    records = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for arxiv_id in sorted(set(arxiv_ids)):
            source = output_root / arxiv_id / "stats.json"
            try:
                stats = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            handle.write(json.dumps(stats, ensure_ascii=False) + "\n")
            records += 1
    os.replace(temporary, destination)
    return records


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
    stats_path = output_root / "m2_stats.json"
    try:
        previous_summary = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_summary = {}
    papers = discover_papers(input_root)
    if args.limit is not None:
        papers = papers[: args.limit]
    versions_requested = sum(len(load_version_specs(paper)[1]) for paper in papers)

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
    freshly_processed = [row for row in results if row["status"] == "ok"]
    completed = [row for row in results if row["status"] in {"ok", "skipped"}]
    skipped = [row for row in results if row["status"] == "skipped"]
    paper_stats_records = _write_paper_stats_jsonl(
        output_root, [str(row["arxiv_id"]) for row in completed]
    )
    last_run = {
        "wall_time_s": round(wall_time, 3),
        "avg_seconds_per_processed_paper": (
            round(
                sum(row["seconds"] for row in freshly_processed) / len(freshly_processed),
                6,
            )
            if freshly_processed
            else None
        ),
        "peak_ram_mb": memory.peak_mb,
        "avg_ram_mb": memory.average_mb,
        "workers": workers,
        "papers_freshly_processed": len(freshly_processed),
        "papers_skipped": len(skipped),
    }
    full_benchmark = len(freshly_processed) == len(papers) and not errors
    previous_benchmark = previous_summary.get("benchmark")
    if not isinstance(previous_benchmark, dict) and previous_summary:
        previous_benchmark = {
            "complete_corpus_run": (
                previous_summary.get("papers_processed")
                == previous_summary.get("papers_requested")
                and previous_summary.get("papers_failed", 0) == 0
            ),
            "wall_time_s": previous_summary.get("wall_time_s"),
            "avg_seconds_per_processed_paper": previous_summary.get(
                "avg_seconds_per_processed_paper"
            ),
            "peak_ram_mb": previous_summary.get("peak_ram_mb"),
            "avg_ram_mb": previous_summary.get("avg_ram_mb"),
            "workers": previous_summary.get("workers"),
            "papers_requested": previous_summary.get("papers_requested"),
            "versions_requested": previous_summary.get("versions_requested"),
        }
    if isinstance(previous_benchmark, dict) and (
        previous_benchmark.get("papers_requested", len(papers)) != len(papers)
        or previous_benchmark.get("versions_requested", versions_requested)
        != versions_requested
    ):
        # A benchmark produced with another --limit value is a different scope.
        previous_benchmark = None
    if full_benchmark or not isinstance(previous_benchmark, dict):
        benchmark = {
            "complete_corpus_run": full_benchmark,
            "papers_requested": len(papers),
            "versions_requested": versions_requested,
            **{
                key: last_run[key]
                for key in (
                    "wall_time_s",
                    "avg_seconds_per_processed_paper",
                    "peak_ram_mb",
                    "avg_ram_mb",
                    "workers",
                )
            },
        }
    else:
        benchmark = previous_benchmark

    versions_parsed = sum(row.get("versions_parsed", 0) for row in completed)
    summary = {
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "papers_requested": len(papers),
        "papers_processed": len(completed),
        "papers_freshly_processed": len(freshly_processed),
        "papers_skipped": len(skipped),
        "papers_failed": len(errors),
        "parse_success_rate": len(completed) / len(papers) if papers else 0.0,
        "versions_requested": versions_requested,
        "version_parse_coverage": (
            versions_parsed / versions_requested
            if versions_requested
            else 0.0
        ),
        "versions_parsed": versions_parsed,
        "empty_versions": sum(row.get("empty_versions", 0) for row in completed),
        "elements_before_dedup": sum(
            row.get("elements_before_dedup", 0) for row in completed
        ),
        "elements_after_dedup": sum(
            row.get("elements_after_dedup", 0) for row in completed
        ),
        "bibtex_entries_before_dedup": sum(
            row.get("bibtex_entries_before_dedup", 0) for row in completed
        ),
        "bibtex_entries_after_dedup": sum(
            row.get("bibtex_entries_after_dedup", 0) for row in completed
        ),
        "unresolved_citation_keys": sum(
            row.get("unresolved_citation_keys", 0) for row in completed
        ),
        "paper_stats_file": "paper_stats.jsonl",
        "paper_stats_records": paper_stats_records,
        # Keep the full fresh-run benchmark stable across resume-only runs.
        "wall_time_s": benchmark.get("wall_time_s"),
        "avg_seconds_per_processed_paper": benchmark.get(
            "avg_seconds_per_processed_paper"
        ),
        "peak_ram_mb": benchmark.get("peak_ram_mb"),
        "avg_ram_mb": benchmark.get("avg_ram_mb"),
        "output_bytes": _directory_size(output_root),
        "workers": benchmark.get("workers"),
        "benchmark": benchmark,
        "last_run": last_run,
    }
    temporary_stats = stats_path.with_suffix(".json.tmp")
    temporary_stats.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_stats, stats_path)
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

