"""Stream M1/M2 into course-schema Parquet and validate model predictions.

No data is uploaded by this module. Five corpus tables can be exported before
the model is ready. Add actual predictions later; no placeholder predictions or
empty sixth table is presented as a completed matching result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from .clean import normalize_arxiv_id
from .data import index_corpora, load_bibtex, load_references, read_m2_schema_version
from .parquet_schema import BASE_TABLES, SCHEMAS, describe_schemas
from .run_m3 import _atomic_text


def json_rows(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{number}")
                yield value


def text(value, *, required=False):
    if value is None:
        if required:
            raise ValueError("Missing required string")
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"Expected a string, got {value!r}")
    return value


def text_list(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
        raise ValueError("Expected a one-level list of strings")
    return list(value)


def canonical(value, *, allow_normalization=False):
    text(value, required=True)
    normalized = normalize_arxiv_id(value)
    if normalized is None or (not allow_normalization and normalized != value):
        raise ValueError(f"Invalid/noncanonical arXiv string ID: {value!r}")
    return normalized


def iso_date(value):
    value = text(value)
    if value:
        date.fromisoformat(value)
    return value


class ShardWriter:
    """Bounded row buffer plus Parquet row groups, capped rows per shard."""

    def __init__(self, directory: Path, schema, *, batch_rows=8192, shard_rows=250000):
        if batch_rows < 1 or shard_rows < 1:
            raise ValueError("Batch/shard row limits must be positive")
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.schema = schema
        self.batch_rows = batch_rows
        self.shard_rows = shard_rows
        self.buffer = []
        self.writer = None
        self.path = None
        self.rows_in_shard = 0
        self.rows = 0
        self.files = []

    def add(self, row):
        if set(row) != set(self.schema.names):
            raise ValueError(f"Unexpected columns for {self.directory.name}")
        for field in self.schema:
            if not field.nullable and row[field.name] is None:
                raise ValueError(f"Null in required {field.name}")
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_rows:
            self.flush()

    def _close_shard(self):
        if self.writer is not None:
            self.writer.close()
            final = self.path.with_suffix(".parquet")
            self.path.replace(final)
            self.files.append(final)
            self.writer = None

    def flush(self):
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        table.validate(full=True)
        offset = 0
        while offset < len(table):
            if self.writer is None:
                self.path = self.directory / f"part-{len(self.files):05d}.incomplete"
                self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd", use_dictionary=True)
                self.rows_in_shard = 0
            count = min(len(table) - offset, self.shard_rows - self.rows_in_shard)
            self.writer.write_table(table.slice(offset, count), row_group_size=self.batch_rows)
            self.rows_in_shard += count
            self.rows += count
            offset += count
            if self.rows_in_shard == self.shard_rows:
                self._close_shard()
        self.buffer.clear()

    def close(self):
        self.flush()
        self._close_shard()
        if not self.files:
            # A genuinely empty source table still has its complete typed schema.
            empty = self.directory / "part-00000.parquet"
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), empty, compression="zstd")
            self.files.append(empty)

    def abort(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def mapped_paper(pid, metadata):
    if canonical(metadata.get("arxiv_id"), allow_normalization=True) != pid:
        raise ValueError(f"Metadata belongs to another publication: {pid}")
    revised = text_list(metadata.get("revised_dates"))
    for value in revised or []:
        iso_date(value)
    return {
        "arxiv_id": pid,
        "title": text(metadata.get("title")),
        "authors": text_list(metadata.get("authors")),
        "submission_date": iso_date(metadata.get("submission_date")),
        "revised_dates": revised,
        "venue": text(metadata.get("venue")),
    }


def iter_parquet(dataset: Path, name: str):
    for path in sorted((dataset / "data" / name).glob("*.parquet")):
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=8192):
            yield from batch.to_pylist()


class PaperCursor:
    """Read one sorted publication group without holding a full table."""

    def __init__(self, iterator):
        self.iterator = iter(iterator)
        self.pending = next(self.iterator, None)

    def take(self, pid):
        if self.pending is not None and self.pending["arxiv_id"] < pid:
            raise ValueError("Rows are unsorted or belong to an unknown publication")
        while self.pending is not None and self.pending["arxiv_id"] == pid:
            row = self.pending
            self.pending = next(self.iterator, None)
            yield row

    def finish(self):
        if self.pending is not None:
            raise ValueError(f"Unknown publication remaining: {self.pending['arxiv_id']}")


def validate_dataset(dataset: Path):
    """Read every exported row and validate types, keys and foreign keys."""
    counts = {}
    expected_names = [name for name in SCHEMAS if (dataset / "data" / name).is_dir()]
    if any(name not in expected_names for name in BASE_TABLES):
        raise ValueError("All five M1/M2 tables must be present")
    for name in expected_names:
        files = sorted((dataset / "data" / name).glob("*.parquet"))
        if not files:
            raise ValueError(f"No Parquet shards for {name}")
        total = 0
        for path in files:
            file = pq.ParquetFile(path)
            if not file.schema_arrow.equals(SCHEMAS[name], check_metadata=False):
                raise ValueError(f"Schema mismatch: {path}\n{file.schema_arrow}")
            for batch in file.iter_batches(batch_size=8192):
                batch.validate(full=True)
                for field, column in zip(batch.schema, batch.columns):
                    if not field.nullable and column.null_count:
                        raise ValueError(f"Required column has null: {name}.{field.name}")
                total += len(batch)
        counts[name] = total

    cursors = {name: PaperCursor(iter_parquet(dataset, name)) for name in expected_names if name != "papers"}
    previous = None
    kinds = Counter()
    partitions = Counter()
    for paper in iter_parquet(dataset, "papers"):
        pid = canonical(paper["arxiv_id"])
        if previous is not None and pid <= previous:
            raise ValueError(f"Duplicate/unsorted publication: {pid}")
        previous = pid
        mapped_paper(pid, paper)
        ref_ids = set()
        for ref in cursors["references"].take(pid):
            ref_id = canonical(ref["ref_arxiv_id"])
            if ref_id in ref_ids:
                raise ValueError(f"Duplicate reference: {pid}/{ref_id}")
            ref_ids.add(ref_id)
            text_list(ref["ref_authors"])
        element_ids = set()
        for element in cursors["elements"].take(pid):
            eid = text(element["element_id"], required=True)
            if eid in element_ids:
                raise ValueError(f"Duplicate element ID: {pid}/{eid}")
            element_ids.add(eid)
            kinds[element["kind"]] += 1
        parent_edges = set()
        for edge in cursors["hierarchy"].take(pid):
            if not isinstance(edge["version"], int) or edge["version"] < 1:
                raise ValueError(f"Invalid hierarchy version: {pid}")
            if edge["element_id"] not in element_ids or edge["parent_id"] not in element_ids:
                raise ValueError(f"Dangling hierarchy edge: {pid}")
            key = (edge["version"], edge["element_id"])
            if key in parent_edges or edge["element_id"] == edge["parent_id"]:
                raise ValueError(f"Duplicate child/self-loop hierarchy edge: {pid}/{key}")
            parent_edges.add(key)
        bib_keys = set()
        for bib in cursors["bibtex"].take(pid):
            key = text(bib["citation_key"], required=True)
            if key in bib_keys:
                raise ValueError(f"Duplicate citation key: {pid}/{key}")
            bib_keys.add(key)
            text_list(bib["authors"])
        if "predictions" in cursors:
            pred_keys = set()
            paper_partitions = set()
            for row in cursors["predictions"].take(pid):
                key = row["citation_key"]
                if key in pred_keys or key not in bib_keys:
                    raise ValueError(f"Duplicate/unknown prediction entry: {pid}/{key}")
                pred_keys.add(key)
                validate_prediction(row, ref_ids)
                partitions[row["partition"]] += 1
                paper_partitions.add(row["partition"])
            if len(paper_partitions) > 1:
                raise ValueError(f"Publication split leakage: {pid}")
            if pred_keys and pred_keys != bib_keys:
                raise ValueError(f"Missing predictions for BibTeX entries of {pid}")
    for cursor in cursors.values():
        cursor.finish()
    return {
        "validated": True,
        "row_counts": counts,
        "element_kinds": dict(kinds),
        "prediction_partitions": dict(partitions),
        "ready_for_final_submission": "predictions" in expected_names and counts.get("predictions", 0) > 0,
        "checks": ["explicit_schemas", "flat_primitive_or_string_list_columns", "required_non_null",
                   "canonical_string_arxiv_ids", "unique_keys", "hierarchy_foreign_keys",
                   "publication_scoped_reference_candidates", "prediction_coverage_if_present"],
    }


def validate_prediction(row, candidates):
    if set(row) != set(SCHEMAS["predictions"].names):
        raise ValueError("Predictions must contain exactly the 9 specified columns")
    canonical(row["arxiv_id"])
    text(row["citation_key"], required=True)
    if row["partition"] not in {"train", "valid", "test"}:
        raise ValueError(f"Invalid prediction partition: {row['partition']!r}")
    truth = row["groundtruth_arxiv_id"]
    if truth is not None and canonical(truth) not in candidates:
        raise ValueError(f"Ground truth is outside publication candidates: {truth}")
    selected = set()
    found_null = False
    for index in range(1, 6):
        value = row[f"cand_{index}"]
        if value is None:
            found_null = True
            continue
        if found_null or canonical(value) not in candidates or value in selected:
            raise ValueError(f"Invalid/duplicate/noncontiguous candidate: {value}")
        selected.add(value)
    if len(selected) != min(5, len(candidates)):
        raise ValueError("Rank all available candidates up to five; null only pads a short candidate pool")


def write_card(dataset: Path, report):
    tables = [name for name in SCHEMAS if name in report["row_counts"]]
    lines = ["---", "pretty_name: Group 8 arXiv corpus and reference matching", "configs:"]
    for name in tables:
        lines.extend([f"- config_name: {name}", "  data_files:", "  - split: train", f"    path: data/{name}/*.parquet"])
    lines.extend(["---", "", "# Group 8 arXiv corpus", "",
                  "Course project: Introduction to Data Science, HCMUS CSC14119.", "",
                  "## Export status", "",
                  "All six tables are present." if report["ready_for_final_submission"] else
                  "DRAFT: five M1/M2 tables are ready; actual model predictions are still pending. This is not a complete M3 submission.",
                  "", "## Tables", "", "| Table | Rows |", "| --- | ---: |"])
    lines.extend(f"| {name} | {report['row_counts'][name]} |" for name in tables)
    lines.extend(["", "## Schema", ""])
    for name in tables:
        columns = "; ".join(f"`{field.name}`: {field.type}" + (" (required)" if not field.nullable else " (nullable)") for field in SCHEMAS[name])
        lines.append(f"- **{name}**: {columns}.")
    lines.extend(["", "## Identifiers and relationships", "",
                  "All arXiv IDs are strings, preserving leading/trailing zeroes and old archive/YYMMNNN IDs. Public publication IDs use dotted notation. Element IDs retain their original M2 identifiers; join on exact element_id/parent_id, not on their prefix.",
                  "", "The arxiv_id in references is the citing publication; ref_arxiv_id is the cited reference. Only references carrying an arXiv ID are published. Paper metadata retains publications with empty source versions. Those versions have no fabricated parsed elements or hierarchy edges.",
                  "", "All five corpus tables cover the full assigned corpus. Predictions cover only publications used in matching. Each table is an HF configuration; the Hub split named train simply contains that table. The predictions.partition column is the actual model train/valid/test assignment.",
                  "", "## Provenance and limitations", "",
                  "Paper metadata and LaTeX sources were collected from arXiv. Reference metadata was collected from Semantic Scholar. M2 parser schema 3 produced the element/hierarchy/BibTeX stores. Source records may contain parsing or metadata errors; schema validation does not establish semantic accuracy.",
                  "", "The reviewed matching labels include AI-assisted annotation (known for Range 4) and human review. Automatic labels are heuristic silver labels. These labels do not constitute an independent purely human benchmark.",
                  "", "Upstream papers can carry different licenses. This generated card deliberately does not assert a blanket CC/MIT license for third-party paper text. The group must follow the instructor's hosting instructions and applicable source terms.",
                  "", "## Loading", "", "```python", "from datasets import load_dataset",
                  "# After the group publishes this dataset, replace REPO_ID with its Hub identifier.",
                  "papers = load_dataset(REPO_ID, 'papers', split='train', streaming=True)",
                  "```", "", "No custom loading script is required.", "", "## Specifications and sources", "",
                  "- https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/crawling/",
                  "- https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/parsing/",
                  "- https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/matching/",
                  "- https://huggingface.co/docs/hub/datasets-manual-configuration", ""])
    if (dataset / "model_selection.json").exists() and (dataset / "matching_metrics.json").exists():
        selection = json.loads((dataset / "model_selection.json").read_text(encoding="utf-8"))
        metrics = json.loads((dataset / "matching_metrics.json").read_text(encoding="utf-8"))
        lines.extend(["## Matching model", "",
                      f"Selected model: `{selection['selected_model']}`. Selection criterion: {selection['criterion']}.", "",
                      "TF-IDF is fitted on training publications only. Publication splits and labels are frozen before model selection. Test is evaluated after selection; it is not used for tuning.", "",
                      "### Validation comparison", "", "| Model | Source-balanced MRR@5 |", "| --- | ---: |"])
        lines.extend(f"| {item['name']} | {item['validation_objective']:.6f} |" for item in selection["comparison"])
        selected = next(item for item in selection["comparison"] if item["name"] == selection["selected_model"])
        baseline = next((item for item in selection["comparison"] if item["name"] == "fixed_combo_baseline"), None)
        if baseline and abs(selected["validation_objective"] - baseline["validation_objective"]) < 1e-12:
            lines.extend(["", "The selected trained model ties the fixed baseline on Validation; no improvement over that baseline is claimed."])
        lines.extend(["", "### Held-out Test", "", metrics["definition"], "",
                      "| Label source | Entries | Matched | No match | MRR@5 (all) | MRR@5 (matched only) |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        groups = dict(metrics["by_partition_source"]["test"])
        groups["combined"] = metrics["by_partition"]["test"]
        for name, values in groups.items():
            lines.append(f"| {name} | {values['entries']} | {values['matched_entries']} | {values['no_match_entries']} | {values['mrr_at_5']:.6f} | {values['mrr_at_5_matched_only']:.6f} |")
        lines.extend(["", *[f"- {item}" for item in metrics["limitations"]], "",
                      "Full reports: `model_selection.json`, `matching_metrics.json`, `model_training_report.json`.", ""])
    _atomic_text(dataset / "README.md", "\n".join(lines))


def export_corpus(args):
    start = time.perf_counter()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Choose a new output folder: {output}")
    if read_m2_schema_version(args.m2) != 3:
        raise ValueError("M2 parser schema 3 is required")
    m1, m2 = index_corpora(args.m1, args.m2)
    if set(m1) != set(m2):
        raise ValueError("M1/M2 publication coverage differs")
    ids = sorted(m1)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No input publications")
    output.mkdir(parents=True)
    _atomic_text(output / "_EXPORT_IN_PROGRESS", "Incomplete until export and validation finish.\n")
    writers = {name: ShardWriter(output / "data" / name, SCHEMAS[name], batch_rows=args.batch_rows, shard_rows=args.shard_rows) for name in BASE_TABLES}
    process = psutil.Process()
    peak = process.memory_info().rss
    empty_papers = empty_versions = skipped_non_arxiv_refs = 0
    try:
        for index, pid in enumerate(ids, 1):
            metadata = json.loads((m1[pid] / "metadata.json").read_text(encoding="utf-8"))
            writers["papers"].add(mapped_paper(pid, metadata))
            source_refs = json.loads((m1[pid] / "references.json").read_text(encoding="utf-8"))
            skipped_non_arxiv_refs += sum(not row.get("arxiv_id") for row in source_refs)
            for row in source_refs:
                if row.get("arxiv_id"):
                    canonical(row["arxiv_id"], allow_normalization=True)
            for ref in load_references(m1[pid], pid):
                writers["references"].add({
                    "arxiv_id": pid, "ref_arxiv_id": ref.ref_arxiv_id,
                    "ref_title": ref.title, "ref_authors": list(ref.authors),
                    "ref_submission_date": ref.submission_date, "ref_s2_id": ref.semantic_scholar_id,
                })
            if not (m2[pid] / "_SUCCESS").is_file():
                raise ValueError(f"M2 paper not completed: {pid}")
            stats = json.loads((m2[pid] / "stats.json").read_text(encoding="utf-8"))
            empty_versions += len(stats.get("empty_versions") or [])
            element_ids = set()
            for row in json_rows(m2[pid] / "elements.jsonl"):
                if canonical(row.get("arxiv_id"), allow_normalization=True) != pid:
                    raise ValueError(f"Element belongs to another publication: {pid}")
                eid = text(row.get("element_id"), required=True)
                if eid in element_ids:
                    raise ValueError(f"Duplicate element ID: {pid}/{eid}")
                element_ids.add(eid)
                writers["elements"].add({
                    "element_id": eid, "arxiv_id": pid,
                    "kind": text(row.get("kind"), required=True), "content": text(row.get("content")),
                })
            if stats.get("elements_after_dedup") != len(element_ids):
                raise ValueError(f"M2 element statistics mismatch: {pid}")
            empty_papers += not element_ids
            seen_children = set()
            edge_count = 0
            for row in json_rows(m2[pid] / "hierarchy.jsonl"):
                if canonical(row.get("arxiv_id"), allow_normalization=True) != pid:
                    raise ValueError(f"Hierarchy belongs to another publication: {pid}")
                version = row.get("version")
                if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                    raise ValueError(f"Invalid version for {pid}: {version!r}")
                eid, parent = row.get("element_id"), row.get("parent_id")
                if eid not in element_ids or parent not in element_ids or eid == parent:
                    raise ValueError(f"Dangling/self-loop hierarchy edge: {pid}")
                key = (version, eid)
                if key in seen_children:
                    raise ValueError(f"Multiple parent rows in one version: {pid}/{key}")
                seen_children.add(key)
                edge_count += 1
                writers["hierarchy"].add({"arxiv_id": pid, "version": version, "element_id": eid, "parent_id": parent})
            if stats.get("hierarchy_edges") != edge_count:
                raise ValueError(f"M2 hierarchy statistics mismatch: {pid}")
            entries = load_bibtex(m2[pid], pid)
            if stats.get("bibtex_entries_after_dedup") != len(entries):
                raise ValueError(f"M2 BibTeX statistics mismatch: {pid}")
            for entry in entries:
                writers["bibtex"].add(asdict(entry))
            peak = max(peak, process.memory_info().rss)
            if index % 500 == 0 or index == len(ids):
                print(f"[{index}/{len(ids)}] papers exported; RSS {process.memory_info().rss / 1024**2:.1f} MiB", flush=True)
        for writer in writers.values():
            writer.close()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    print("Reading all Parquet shards to verify schema, keys and relationships...", flush=True)
    report = validate_dataset(output)
    expected = {name: writer.rows for name, writer in writers.items()}
    if report["row_counts"] != expected:
        raise ValueError("Written/readback row counts do not match")
    report.update({
        "exported_publications": len(ids), "limited_export": args.limit is not None,
        "empty_parsed_publications": empty_papers, "empty_source_versions": empty_versions,
        "references_without_arxiv_excluded": skipped_non_arxiv_refs,
        "wall_time_s": round(time.perf_counter() - start, 3),
        "peak_sampled_rss_mib": round(max(peak, process.memory_info().rss) / 1024**2, 3),
        "status": "five_tables_validated_predictions_pending",
        "parquet_bytes": sum(p.stat().st_size for p in (output / "data").rglob("*.parquet")),
        "shards": {name: len(writer.files) for name, writer in writers.items()},
        "m2_parser_schema_version": 3,
    })
    _atomic_text(output / "schema.json", json.dumps(describe_schemas(), indent=2) + "\n")
    _atomic_text(output / "export_report.json", json.dumps(report, indent=2) + "\n")
    write_card(output, report)
    (output / "_EXPORT_IN_PROGRESS").unlink()
    _atomic_text(output / "_BASE_TABLES_SUCCESS", "Five source tables validated; predictions still pending.\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def prediction_input(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield {key: (None if value == "" else value) for key, value in row.items()}
    elif path.suffix.lower() == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            yield from batch.to_pylist()
    else:
        yield from json_rows(path)


def add_predictions(args):
    dataset = args.dataset
    if not (dataset / "_BASE_TABLES_SUCCESS").is_file():
        raise ValueError("A successfully validated base export is required")
    previous = json.loads((dataset / "export_report.json").read_text(encoding="utf-8"))
    if previous.get("limited_export"):
        raise ValueError("A --limit smoke export cannot become the final dataset")
    target = dataset / "data" / "predictions"
    if target.exists():
        raise FileExistsError("Predictions already exist; use a new dataset copy to replace them")
    partitions = {}
    for row in json_rows(args.partitions):
        pid = canonical(row["arxiv_id"])
        if pid in partitions or row.get("partition") not in {"train", "valid", "test"}:
            raise ValueError("Duplicate/invalid partition assignment")
        partitions[pid] = row["partition"]
    truth = {}
    for row in json_rows(args.labels):
        key = (canonical(row["arxiv_id"]), text(row["citation_key"], required=True))
        if key in truth or key[0] not in partitions:
            raise ValueError("Duplicate label or label outside partition pool")
        gt = row.get("groundtruth_arxiv_id")
        truth[key] = canonical(gt) if gt is not None else None
    if {pid for pid, _ in truth} != set(partitions):
        raise ValueError("Label and partition publication sets differ")
    # Keep only the modest matching pool in memory, not the full 25k corpus.
    refs = {pid: set() for pid in partitions}
    for row in iter_parquet(dataset, "references"):
        if row["arxiv_id"] in refs:
            refs[row["arxiv_id"]].add(row["ref_arxiv_id"])
    expected_keys = {(row["arxiv_id"], row["citation_key"]) for row in iter_parquet(dataset, "bibtex") if row["arxiv_id"] in partitions}
    if set(truth) != expected_keys:
        raise ValueError("For final predictions, provide every BibTeX entry of all matching publications")
    rows = []
    seen = set()
    for row in prediction_input(args.input):
        key = (row.get("arxiv_id"), row.get("citation_key"))
        if key in seen or key not in truth:
            raise ValueError(f"Duplicate/unknown prediction key: {key}")
        seen.add(key)
        validate_prediction(row, refs[key[0]])
        if row["partition"] != partitions[key[0]] or row["groundtruth_arxiv_id"] != truth[key]:
            raise ValueError(f"Prediction partition/ground truth differs from frozen labels: {key}")
        rows.append(row)
    if seen != expected_keys:
        raise ValueError(f"Prediction coverage mismatch: missing {len(expected_keys - seen)} entries")
    if not rows:
        raise ValueError("No actual model predictions supplied")
    writer = ShardWriter(target, SCHEMAS["predictions"], batch_rows=args.batch_rows, shard_rows=args.shard_rows)
    try:
        for row in sorted(rows, key=lambda r: (r["arxiv_id"], r["citation_key"])):
            writer.add(row)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    report = validate_dataset(dataset)
    previous.update(report)
    previous["status"] = "six_tables_validated"
    previous["prediction_input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    previous["parquet_bytes"] = sum(p.stat().st_size for p in (dataset / "data").rglob("*.parquet"))
    previous["shards"]["predictions"] = len(writer.files)
    _atomic_text(dataset / "export_report.json", json.dumps(previous, indent=2) + "\n")
    write_card(dataset, previous)
    _atomic_text(dataset / "_FINAL_DATASET_SUCCESS", "All six tables validated with supplied model predictions.\n")
    print(json.dumps(previous, ensure_ascii=False, indent=2))
    return 0


def document_model(args):
    """Attach reports only when they describe the predictions actually exported."""
    report = json.loads((args.dataset / "export_report.json").read_text(encoding="utf-8"))
    if not report.get("ready_for_final_submission"):
        raise ValueError("Add and validate all six tables before documenting the model")
    predictions = args.model_run / "predictions.jsonl"
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != report.get("prediction_input_sha256"):
        raise ValueError("Model-run predictions do not match the exported predictions")
    sources = {"selection.json": "model_selection.json", "metrics.json": "matching_metrics.json",
               "training_report.json": "model_training_report.json"}
    contents = {target: json.loads((args.model_run / source).read_text(encoding="utf-8"))
                for source, target in sources.items()}
    if contents["model_selection.json"]["selected_model"] != contents["matching_metrics.json"]["selected_model"]:
        raise ValueError("Model selection and metric reports disagree")
    for name, content in contents.items():
        _atomic_text(args.dataset / name, json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    write_card(args.dataset, report)
    print("Documented model and matching metrics in dataset card")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Export the five full-corpus M1/M2 tables")
    export.add_argument("--m1", type=Path, required=True)
    export.add_argument("--m2", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--limit", type=int, help="Smoke test only; omit for final export")
    export.set_defaults(func=export_corpus)
    predictions = sub.add_parser("add-predictions", help="Validate and add actual model top-5 output")
    predictions.add_argument("--dataset", type=Path, required=True)
    predictions.add_argument("--input", type=Path, required=True)
    predictions.add_argument("--labels", type=Path, required=True)
    predictions.add_argument("--partitions", type=Path, required=True)
    predictions.set_defaults(func=add_predictions)
    document = sub.add_parser("document-model", help="Attach model reports matching the exported predictions")
    document.add_argument("--dataset", type=Path, required=True)
    document.add_argument("--model-run", type=Path, required=True)
    document.set_defaults(func=document_model)
    check = sub.add_parser("validate", help="Read back every table and verify keys and schemas")
    check.add_argument("--dataset", type=Path, required=True)
    check.set_defaults(func=lambda args: print(json.dumps(validate_dataset(args.dataset), ensure_ascii=False, indent=2)) or 0)
    for command in (export, predictions):
        command.add_argument("--batch-rows", type=int, default=8192)
        command.add_argument("--shard-rows", type=int, default=250000)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
