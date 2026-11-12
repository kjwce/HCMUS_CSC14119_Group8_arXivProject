"""Split a validated six-table corpus by original member manifests, without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .export_parquet import canonical, json_rows, validate_dataset
from .parquet_schema import SCHEMAS, describe_schemas
from .run_m3 import _atomic_text


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_assignments(specs, expected_per_range=5000):
    """Treat the manifests, rather than numeric bounds or reference IDs, as ownership."""
    if expected_per_range < 1:
        raise ValueError("Expected range size must be positive")
    assignments, provenance, owner = {}, {}, {}
    for spec in specs:
        name, separator, filename = spec.partition("=")
        if not separator or not re.fullmatch(r"r[1-9][0-9]*", name) or name in assignments:
            raise ValueError(f"Expected a unique rN=manifest.jsonl specification: {spec}")
        path = Path(filename)
        ids = []
        for row in json_rows(path):
            pid = canonical(row["arxiv_id"], allow_normalization=True)
            if pid in owner:
                raise ValueError(f"Duplicate/overlapping manifest paper: {pid}")
            owner[pid] = name
            ids.append(pid)
        if len(ids) != expected_per_range:
            raise ValueError(f"{name}: expected {expected_per_range} assigned papers, got {len(ids)}")
        assignments[name] = set(ids)
        provenance[name] = {"manifest_filename": path.name, "manifest_sha256": digest(path)}
    if not assignments:
        raise ValueError("No range manifests supplied")
    return assignments, provenance, owner


def batches(dataset, name):
    files = sorted((Path(dataset) / "data" / name).glob("*.parquet"))
    if not files:
        raise ValueError(f"Missing Parquet table: {name}")
    for path in files:
        reader = pq.ParquetFile(path)
        try:
            if not reader.schema_arrow.equals(SCHEMAS[name], check_metadata=False):
                raise ValueError(f"Schema mismatch: {path}")
            yield from reader.iter_batches(batch_size=8192)
        finally:
            reader.close()


class BatchWriter:
    """Preserve Arrow values and schemas while bounding row groups and shard size."""

    def __init__(self, directory, schema, shard_rows):
        self.directory, self.schema = directory, schema
        self.shard_rows = shard_rows
        self.directory.mkdir(parents=True, exist_ok=False)
        self.writer = None
        self.files = []
        self.rows = self.current_rows = 0

    def finish_shard(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            final = self.path.with_suffix(".parquet")
            self.path.replace(final)
            self.files.append(final)

    def write(self, batch):
        if not batch.schema.equals(self.schema, check_metadata=False):
            raise ValueError("Unexpected batch schema")
        offset = 0
        while offset < len(batch):
            if self.writer is None:
                self.path = self.directory / f"part-{len(self.files):05d}.incomplete"
                self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd", use_dictionary=True)
                self.current_rows = 0
            count = min(len(batch) - offset, self.shard_rows - self.current_rows)
            self.writer.write_batch(batch.slice(offset, count), row_group_size=8192)
            self.current_rows += count
            self.rows += count
            offset += count
            if self.current_rows == self.shard_rows:
                self.finish_shard()

    def close(self):
        self.finish_shard()
        if not self.files:
            path = self.directory / "part-00000.parquet"
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), path, compression="zstd")
            self.files.append(path)

    def abort(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def plain(batch):
    # Parquet can legitimately change dictionary encodings without changing values.
    return pa.RecordBatch.from_arrays(
        [column.dictionary_decode() if pa.types.is_dictionary(column.type) else column
         for column in batch.columns], names=batch.schema.names,
    )


class ValueCursor:
    """Compare values independently of Parquet shard and row-group boundaries."""

    def __init__(self, iterator):
        self.iterator = iter(iterator)
        self.current = next(self.iterator, None)
        self.offset = 0
        self.rows = 0

    def compare(self, expected):
        offset = 0
        while offset < len(expected):
            if self.current is None:
                raise ValueError("Split lost source rows")
            count = min(len(expected) - offset, len(self.current) - self.offset)
            if not plain(expected.slice(offset, count)).equals(plain(self.current.slice(self.offset, count))):
                raise ValueError("Split changed source row values or ordering")
            offset += count
            self.offset += count
            self.rows += count
            if self.offset == len(self.current):
                self.current = next(self.iterator, None)
                self.offset = 0

    def finish(self):
        if self.current is not None:
            raise ValueError("Split contains extra rows")


def verify_projection(source, outputs, assignments):
    """Read every output value against an independent Arrow membership filter."""
    totals = {name: {} for name in outputs}
    values = {name: pa.array(sorted(ids), type=pa.string()) for name, ids in assignments.items()}
    for table in SCHEMAS:
        cursors = {name: ValueCursor(batches(path, table)) for name, path in outputs.items()}
        for batch in batches(source, table):
            ids = batch.column(batch.schema.get_field_index("arxiv_id"))
            for name, cursor in cursors.items():
                cursor.compare(batch.filter(pc.is_in(ids, value_set=values[name])))
        for name, cursor in cursors.items():
            cursor.finish()
            totals[name][table] = cursor.rows
        print(f"Verified every original row value: {table}", flush=True)
    return totals


def write_card(output, name, ids, report, source_repo):
    lines = ["---", f"pretty_name: Group 8 arXiv corpus - {name}", "configs:"]
    for table in SCHEMAS:
        lines += [f"- config_name: {table}", "  data_files:", "  - split: train", f"    path: data/{table}/*.parquet"]
    lines += ["---", "", f"# Group 8 arXiv corpus - {name}", "",
              "HCMUS CSC14119 Introduction to Data Science course project.", "",
              "## Individual submission scope", "",
              f"This dataset contains the {len(ids):,} publications assigned to {name} in the original M1 manifest. "
              f"The smallest/largest assigned IDs are `{min(ids)}` and `{max(ids)}`. "
              "Membership is determined by the full manifest, not by assuming a continuous numeric interval.", "",
              "All six tables are filtered by the citing/publication `arxiv_id`. In references, "
              "`ref_arxiv_id` identifies the cited work and may belong to another member's range or lie outside all assigned ranges. "
              "Those references are retained. Original string IDs, trailing zeroes, element identifiers and hierarchy edges are preserved.", "",
              "## Tables", "", "| Configuration | Rows |", "| --- | ---: |"]
    lines += [f"| {table} | {report['row_counts'][table]:,} |" for table in SCHEMAS]
    lines += ["", "## Matching experiment", "",
              "The group trained and evaluated one shared model. These predictions are the unchanged subset "
              "of that completed experiment belonging to this range. No model was retrained and no ground truth, "
              "candidate ranking or train/valid/test assignment was changed during this storage split.", "",
              "The Hub split named `train` stores the complete table in each configuration. "
              "The actual modelling partition is the `predictions.partition` column. "
              "The group-level split was drawn across all selected publications; individual range subsets "
              "can have unequal holdout counts, including zero rows in a particular partition.", "",
              "| Modelling partition | Prediction rows |", "| --- | ---: |"]
    lines += [f"| {part} | {report['prediction_partitions'].get(part, 0):,} |" for part in ("train", "valid", "test")]
    lines += ["", "## Provenance and limitations", "",
              f"Full group corpus and group-level model reports: https://huggingface.co/datasets/{source_repo}", "",
              "Source metadata and LaTeX were collected from arXiv; reference metadata came from Semantic Scholar. "
              "The five corpus tables cover all assigned publications, including papers with empty/unavailable TeX versions. "
              "Predictions cover the publications selected for matching, not every assigned paper.", "",
              "Automatic matching labels are heuristic silver labels. Reviewed manual labels include AI assistance "
              "and human review, and are not an independent purely human benchmark. Group model scores are not "
              "individual-range scores. Schema and projection checks do not establish semantic label correctness.", "",
              "Papers may carry different upstream licenses; this card does not assert a blanket license for third-party text.", "",
              "`assigned_papers.jsonl` lists the exact membership. `range_report.json` records manifest/source hashes "
              "and full-value projection verification; `export_report.json` records per-range schema, keys, "
              "hierarchy and reference-candidate checks. `schema.json` documents all column types.", "",
              "## Loading", "", "After uploading to your personal public dataset repository, replace YOUR_USERNAME/REPO_NAME:", "",
              "```python", "from datasets import load_dataset",
              'dataset = load_dataset("YOUR_USERNAME/REPO_NAME", "papers", split="train", streaming=True)',
              "print(next(iter(dataset)))", "```", "",
              "The other configurations are references, elements, hierarchy, bibtex and predictions. "
              "No custom loading script is needed.", ""]
    _atomic_text(output / "README.md", "\n".join(lines))


def upload_guide(name):
    return f"""# Hướng dẫn upload bộ {name}

1. Giải nén ZIP. Mở thư mục `{name.replace('r', 'range_', 1)}_dataset` có `README.md` và `data/`.
2. Đăng nhập tài khoản Hugging Face CÁ NHÂN, tạo một dataset Public.
3. Upload NỘI DUNG thư mục dataset, giữ `README.md` ở gốc và `data/` cùng cấp.
   Không upload nguyên file ZIP làm dataset, không thêm một lớp thư mục bọc ngoài.
4. Giữ đầy đủ 6 thư mục bảng trong `data/`, kể cả khi một bảng có nhiều shard.
   Có thể upload thêm `schema.json`, `export_report.json`, `range_report.json`, `assigned_papers.jsonl`.
5. Đổi `YOUR_USERNAME/REPO_NAME` trong ví dụ đọc ở README thành repo của bạn.
6. Kiểm tra 6 config trên Dataset Viewer, rồi nộp link public của repo cá nhân lên Moodle.

Nếu upload bằng CLI (cài `huggingface_hub` trước), mở terminal trong thư mục dataset:

```text
hf auth login
hf upload YOUR_USERNAME/REPO_NAME . . --repo-type dataset --include "data/*/*.parquet" --include "README.md" --include "schema.json" --include "export_report.json" --include "range_report.json" --include "assigned_papers.jsonl"
```

Chỉ đăng nhập trên máy của bạn; không gửi token vào chat hay đưa vào file.
Model và partition giữ nguyên từ lần chạy chung của nhóm. Một range có thể không
có dòng Test/Validation vì nhóm chia tập trên pool chung; không tự đổi partition.
Mỗi bộ có 5.000 dòng papers; các bảng còn lại có số dòng khác nhau theo nội dung.
"""


def split_dataset(args):
    started = time.perf_counter()
    source, output = args.dataset.resolve(), args.output.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Choose a new output folder outside the source dataset")
    if args.shard_rows < 1:
        raise ValueError("--shard-rows must be positive")
    assignments, provenance, owner = load_assignments(args.range_manifest, args.expected_per_range)
    if len(assignments) != 5:
        raise ValueError("Provide the five original member manifests")
    source_report = json.loads((source / "export_report.json").read_text(encoding="utf-8"))
    if not source_report.get("ready_for_final_submission") or source_report.get("limited_export"):
        raise ValueError("Source must be the completed full six-table dataset")
    paper_ids = [pid for batch in batches(source, "papers") for pid in batch.column("arxiv_id").to_pylist()]
    if len(paper_ids) != len(set(paper_ids)) or set(paper_ids) != set(owner):
        raise ValueError("Manifest union does not exactly cover the source papers")
    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    source_hashes = {path.relative_to(source).as_posix(): digest(path) for path in source_files}
    source_bytes = sum(path.stat().st_size for path in source_files)
    if shutil.disk_usage(output.parent).free < source_bytes * 3:
        raise ValueError("Not enough free space for range datasets and ZIP packages")
    output.mkdir(parents=True)
    _atomic_text(output / "_IN_PROGRESS", "Splitting and verifying the original dataset.\n")
    outputs = {name: output / f"{name.replace('r', 'range_', 1)}_dataset" for name in sorted(assignments)}
    for table, schema in SCHEMAS.items():
        writers = {name: BatchWriter(path / "data" / table, schema, args.shard_rows) for name, path in outputs.items()}
        try:
            for batch in batches(source, table):
                indices = {name: [] for name in outputs}
                for number, pid in enumerate(batch.column("arxiv_id").to_pylist()):
                    if pid not in owner:
                        raise ValueError(f"Row outside assigned corpus: {table}/{pid}")
                    indices[owner[pid]].append(number)
                for name, selected in indices.items():
                    if selected:
                        writers[name].write(batch.take(pa.array(selected, type=pa.int32())))
            for writer in writers.values():
                writer.close()
            counts = {name: writer.rows for name, writer in writers.items()}
            if sum(counts.values()) != source_report["row_counts"][table]:
                raise ValueError(f"Source count mismatch: {table}")
            print(f"Split {table}: {counts}", flush=True)
        finally:
            for writer in writers.values():
                writer.abort()
    verified = verify_projection(source, outputs, assignments)
    reports = {}
    for name, path in outputs.items():
        print(f"Validating keys, types, hierarchy and candidates: {name}", flush=True)
        report = validate_dataset(path)
        if report["row_counts"] != verified[name] or not report["ready_for_final_submission"]:
            raise ValueError(f"Incomplete range dataset: {name}")
        report.update({"status": "six_tables_validated", "scope": name, "limited_export": False,
                       "exported_publications": len(assignments[name]),
                       "parquet_bytes": sum(p.stat().st_size for p in (path / "data").rglob("*.parquet"))})
        range_report = {"range": name, "assigned_publications": len(assignments[name]),
                        "minimum_id": min(assignments[name]), "maximum_id": max(assignments[name]),
                        **provenance[name], "source_dataset": args.source_repo,
                        "source_sha256": source_hashes, "projection": "arxiv_id membership in original M1 manifest",
                        "all_row_values_and_order_verified": True, "model_retrained": False,
                        "predictions_and_partitions_unchanged": True, "row_counts": report["row_counts"]}
        _atomic_text(path / "schema.json", json.dumps(describe_schemas(), indent=2) + "\n")
        _atomic_text(path / "export_report.json", json.dumps(report, indent=2) + "\n")
        _atomic_text(path / "range_report.json", json.dumps(range_report, indent=2) + "\n")
        _atomic_text(path / "assigned_papers.jsonl", "".join(json.dumps({"arxiv_id": pid}) + "\n" for pid in sorted(assignments[name])))
        write_card(path, name, assignments[name], report, args.source_repo)
        reports[name] = {"directory": path.name, "assigned_publications": len(assignments[name]),
                         "minimum_id": min(assignments[name]), "maximum_id": max(assignments[name]),
                         "row_counts": report["row_counts"], "prediction_partitions": report["prediction_partitions"]}
    for path in source_files:
        if digest(path) != source_hashes[path.relative_to(source).as_posix()]:
            raise ValueError(f"Source dataset changed during split: {path.name}")
    packages = output / "packages"
    packages.mkdir()
    for name, path in outputs.items():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        zip_path = packages / f"{path.name}.zip"
        with ZipFile(zip_path, "x") as archive:
            for file in files:
                archive.write(file, f"{path.name}/{file.relative_to(path).as_posix()}",
                              compress_type=ZIP_STORED if file.suffix == ".parquet" else ZIP_DEFLATED)
            archive.writestr("HUONG_DAN_UPLOAD.md", upload_guide(name), compress_type=ZIP_DEFLATED)
        with ZipFile(zip_path) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(files) + 1:
                raise ValueError(f"ZIP integrity failed: {zip_path.name}")
        reports[name].update({"zip": f"packages/{zip_path.name}", "zip_bytes": zip_path.stat().st_size,
                              "zip_sha256": digest(zip_path)})
        print(f"Packaged and CRC-checked: {zip_path.name}", flush=True)
    summary = {"status": "complete", "total_publications": len(owner), "ranges": reports,
               "source_unchanged": True, "source_sha256": source_hashes,
               "wall_time_s": round(time.perf_counter() - started, 3)}
    _atomic_text(output / "split_report.json", json.dumps(summary, indent=2) + "\n")
    lines = ["# Gói dữ liệu cá nhân gửi nhóm", "",
             "Gửi mỗi thành viên đúng file ZIP trong packages/ theo range được giao.", "",
             "| Range | ID nhỏ nhất | ID lớn nhất | Paper | File ZIP |", "| --- | --- | --- | ---: | --- |"]
    lines += [f"| {name} | {row['minimum_id']} | {row['maximum_id']} | {row['assigned_publications']} | {row['zip']} |" for name, row in reports.items()]
    lines += ["", "Range 1 đi qua ranh giới tháng; danh sách manifest mới là căn cứ chính xác, không tự suy ra ID bằng phép cộng.",
              "", "Mỗi ZIP có dataset đủ 6 bảng, README cấu hình Hugging Face và hướng dẫn upload cá nhân.",
              "Bộ gốc 25.000 paper không bị sửa. Toàn bộ giá trị của các dòng sau khi tách đã được đối chiếu với bản gốc.",
              "Model, nhãn, top-5 và partition giữ nguyên. Có range không có dòng Test hoặc Validation vì nhóm chia tập trên pool chung.", ""]
    _atomic_text(packages / "GUI_NHOM.md", "\n".join(lines))
    _atomic_text(output / "_SPLIT_SUCCESS", "Five range datasets validated, source unchanged, ZIPs checked.\n")
    (output / "_IN_PROGRESS").unlink()
    print(json.dumps({"output": str(output), "ranges": reports}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--range-manifest", action="append", required=True, help="r1=path/to/original/manifest/papers.jsonl; repeat for five members")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-per-range", type=int, default=5000)
    parser.add_argument("--shard-rows", type=int, default=250000)
    parser.add_argument("--source-repo", default="hngx1016/HCMUS_CSC14119_Group8_arxiv")
    split_dataset(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
