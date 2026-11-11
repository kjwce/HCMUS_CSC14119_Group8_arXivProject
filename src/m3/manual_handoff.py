"""Validate reviewed CSV labels, calibrate on manual-train, and export model inputs.

Run with ``python -m src.m3.manual_handoff --help``. Outputs belong outside
the source repository. The original automatic pool and candidate scores are
reused; only the acceptance thresholds are calibrated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .clean import normalize_arxiv_id, normalize_doi
from .data import index_corpora, load_joined_paper, read_m2_schema_version
from .matcher import label_publication
from .run_m3 import _atomic_jsonl, _atomic_text, _read_jsonl_objects
from .split import split_publications
from .validation import validate_automatic_labels


BASELINE = (0.68, 0.05)


def validate_label_row(row: dict[str, str], candidate_ids: set[str]) -> str | None:
    """Require canonical string IDs; never guess digits stripped by Excel."""
    status = row.get("match_status", "")
    value = row.get("groundtruth_arxiv_id", "")
    if status not in {"matched", "no_match"}:
        raise ValueError(f"Unreviewed/invalid match_status: {status!r}")
    if status == "no_match":
        if value:
            raise ValueError("no_match must have an empty groundtruth_arxiv_id")
        return None
    canonical = normalize_arxiv_id(value)
    if not canonical or canonical != value or value not in candidate_ids:
        raise ValueError(f"Ground truth is not a canonical candidate of this paper: {value!r}")
    return value


def import_reviewed_labels(entries_dir: Path, m1_index, m2_index):
    labels = []
    pool = []
    papers_seen: set[str] = set()
    for range_number in range(1, 6):
        range_name = f"r{range_number}"
        source = entries_dir / f"{range_name}_manual_entries.csv"
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "range", "arxiv_id", "citation_key", "bib_title", "bib_authors",
                "bib_year", "bib_doi", "bib_journal", "bib_url", "match_status",
                "groundtruth_arxiv_id", "annotator_notes",
            }
            if set(reader.fieldnames or []) != required:
                raise ValueError(f"Unexpected CSV columns in {source}")
            by_paper = defaultdict(list)
            for line_number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row at {source}:{line_number}")
                pid = normalize_arxiv_id(row["arxiv_id"])
                if not pid or pid != row["arxiv_id"] or row["range"] != range_name:
                    raise ValueError(f"Wrong range/publication at {source}:{line_number}")
                by_paper[pid].append((line_number, row))
        if len(by_paper) != 5:
            raise ValueError(f"{range_name} must contain exactly 5 papers")
        for pid, source_rows in sorted(by_paper.items()):
            if pid in papers_seen:
                raise ValueError(f"Paper occurs in multiple ranges: {pid}")
            papers_seen.add(pid)
            entries, references = load_joined_paper(pid, m1_index, m2_index)
            bib = {entry.citation_key: entry for entry in entries}
            candidates = {ref.ref_arxiv_id for ref in references}
            keys = set()
            if len(source_rows) < 20:
                raise ValueError(f"Fewer than 20 reviewed labels for {pid}")
            for line_number, row in source_rows:
                key = row["citation_key"]
                if key in keys or key not in bib:
                    raise ValueError(f"Duplicate/unknown citation at {source}:{line_number}: {key}")
                keys.add(key)
                entry = bib[key]
                expected = {
                    "bib_title": entry.title or "",
                    "bib_authors": " | ".join(entry.authors),
                    "bib_year": str(entry.year) if entry.year is not None else "",
                    "bib_journal": entry.journal or "",
                    "bib_url": entry.url or "",
                }
                if any(row[column] != value for column, value in expected.items()):
                    raise ValueError(f"Original BibTeX metadata changed at {source}:{line_number}")
                if normalize_doi(row["bib_doi"]) != entry.doi:
                    raise ValueError(f"Original DOI changed at {source}:{line_number}")
                groundtruth = validate_label_row(row, candidates)
                labels.append({
                    "arxiv_id": pid,
                    "citation_key": key,
                    "groundtruth_arxiv_id": groundtruth,
                    "match_status": row["match_status"],
                    "label_source": "manual",
                    "range": range_name,
                    "source_file": source.name,
                    "source_row": line_number,
                    "annotator_notes": row["annotator_notes"],
                })
            metadata_path = m1_index[pid] / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            pool.append({
                "arxiv_id": pid,
                "range": range_name,
                "title": metadata.get("title"),
                "bibtex_entries": len(entries),
                "arxiv_references": len(references),
                "reviewed_labels": len(source_rows),
                "selection_source": "reviewed_csv_submission",
            })
    return sorted(labels, key=lambda x: (x["arxiv_id"], x["citation_key"])), pool


def prediction(row, threshold: float, margin: float):
    best = row.get("best_candidate_features")
    if not best:
        return None
    if row["label_method"] == "embedded_arxiv_id":
        return best["ref_arxiv_id"]
    if row["confidence"] >= threshold and row["margin"] >= margin:
        return best["ref_arxiv_id"]
    return None


def evaluate(samples, threshold: float, margin: float):
    exact = correct_matches = accepted = actual_matches = 0
    for sample in samples:
        predicted = prediction(sample, threshold, margin)
        truth = sample["reviewed_groundtruth"]
        exact += predicted == truth
        actual_matches += truth is not None
        accepted += predicted is not None
        correct_matches += predicted is not None and predicted == truth
    return {
        "entries": len(samples),
        "exact_correct": exact,
        "exact_accuracy": exact / len(samples) if samples else 0.0,
        "accepted_matches": accepted,
        "correct_matches": correct_matches,
        "actual_matches": actual_matches,
        "match_precision": correct_matches / accepted if accepted else 0.0,
        "match_recall": correct_matches / actual_matches if actual_matches else 0.0,
    }


def calibrate(train_samples, min_precision: float = 0.98):
    """Choose acceptance thresholds using only the passed manual-train rows.

    Prefer maximum correct-match recall subject to 98% training precision;
    ties prefer exact accuracy and then proximity to the old configuration.
    No scoring weights, manual labels, or held-out labels are modified.
    """
    if not train_samples or not any(x["reviewed_groundtruth"] is not None for x in train_samples):
        raise ValueError("Calibration requires labelled manual-train matches")
    thresholds = sorted({round(x / 100, 2) for x in range(50, 91, 2)} | {BASELINE[0]})
    margins = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    results = []
    for threshold in thresholds:
        for margin in margins:
            metrics = evaluate(train_samples, threshold, margin)
            results.append({"threshold": threshold, "minimum_margin": margin, **metrics})
    feasible = [r for r in results if r["accepted_matches"] and r["match_precision"] >= min_precision]
    if not feasible:
        chosen = next(r for r in results if (r["threshold"], r["minimum_margin"]) == BASELINE)
        reason = "No setting met precision constraint; retain baseline."
    else:
        chosen = max(feasible, key=lambda r: (
            r["correct_matches"], r["exact_correct"],
            -abs(r["threshold"] - BASELINE[0]) - abs(r["minimum_margin"] - BASELINE[1]),
            r["threshold"], r["minimum_margin"],
        ))
        reason = "Maximize correct-match recall with manual-train precision >= requested constraint."
    return {"selected": chosen, "minimum_train_precision": min_precision, "selection_reason": reason, "grid": results}


def build(args):
    output = args.output
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if read_m2_schema_version(args.m2) != 3:
        raise ValueError("M2 parser schema 3 is required")
    m1_index, m2_index = index_corpora(args.m1, args.m2)
    manual, pool = import_reviewed_labels(args.entries_dir, m1_index, m2_index)
    automatic_pool = _read_jsonl_objects(args.automatic_pool)
    automatic = _read_jsonl_objects(args.automatic_labels)
    integrity = validate_automatic_labels(automatic_pool, automatic, m1_index, m2_index)
    manual_ids = {x["arxiv_id"] for x in manual}
    automatic_ids = {x["arxiv_id"] for x in automatic_pool}
    if len(automatic_ids) < (len(m1_index.keys() & m2_index.keys()) - len(manual_ids) + 9) // 10:
        raise ValueError("Automatic pool does not cover at least 10% of remaining papers")
    partitions = split_publications(manual_ids, automatic_ids, seed=args.seed)
    partition_map = {x["arxiv_id"]: x["partition"] for x in partitions}
    manual_lookup = {(x["arxiv_id"], x["citation_key"]): x for x in manual}
    samples = defaultdict(list)
    for pid in sorted(manual_ids):
        entries, refs = load_joined_paper(pid, m1_index, m2_index)
        scored = label_publication(entries, refs)
        for row in scored:
            label = manual_lookup.get((pid, row["citation_key"]))
            if label is not None:
                row["reviewed_groundtruth"] = label["groundtruth_arxiv_id"]
                samples[partition_map[pid]].append(row)
    calibration = calibrate(samples["train"])
    threshold = calibration["selected"]["threshold"]
    margin = calibration["selected"]["minimum_margin"]
    report = {
        "seed": args.seed,
        "calibration_source": "manual/train only",
        "train_publications": sorted(x for x in manual_ids if partition_map[x] == "train"),
        "baseline": {"threshold": BASELINE[0], "minimum_margin": BASELINE[1]},
        **calibration,
        "diagnostics": {
            name: {
                "baseline": evaluate(samples[name], *BASELINE),
                "selected": evaluate(samples[name], threshold, margin),
            } for name in ("train", "valid", "test")
        },
        "limitations": [
            "Manual validity/test labels are not used to select acceptance thresholds.",
            "Manual diagnostics are agreement with reviewed submitted labels, not model MRR@5.",
            "Labels in the manual pool include AI-assisted annotation (known for range r4); human review is still needed for an independent manual benchmark.",
            "Automatic labels are heuristic silver labels, not independent ground truth.",
        ],
    }
    changes = 0
    for row in automatic:
        new_groundtruth = prediction(row, threshold, margin)
        changes += new_groundtruth != row["groundtruth_arxiv_id"]
        row["groundtruth_arxiv_id"] = new_groundtruth
        row["accepted"] = new_groundtruth is not None
        row["match_status"] = "matched" if row["accepted"] else "no_match"
        row["calibration_source"] = "manual/train"
        row["calibration_threshold"] = threshold
        row["calibration_minimum_margin"] = margin
    report["automatic_labels_changed"] = changes
    integrity_after = validate_automatic_labels(automatic_pool, automatic, m1_index, m2_index)
    output.mkdir(parents=True)
    _atomic_jsonl(output / "manual_pool.jsonl", sorted(pool, key=lambda x: (x["range"], x["arxiv_id"])))
    _atomic_jsonl(output / "manual_labels.jsonl", manual)
    _atomic_jsonl(output / "automatic_pool.jsonl", automatic_pool)
    _atomic_jsonl(output / "automatic_labels.jsonl", automatic)
    _atomic_jsonl(output / "partitions.jsonl", partitions)
    _atomic_text(output / "matcher_calibration.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    label_fields = ("arxiv_id", "citation_key", "groundtruth_arxiv_id", "label_source", "match_status")
    combined = [
        {**{key: row[key] for key in label_fields}, "partition": partition_map[row["arxiv_id"]]}
        for row in manual + automatic
    ]
    _atomic_jsonl(output / "labels.jsonl", combined)
    for partition in ("train", "valid", "test"):
        _atomic_jsonl(output / f"labels_{partition}.jsonl", (row for row in combined if row["partition"] == partition))

    # Write one publication at a time: do not construct a corpus-sized Cartesian product.
    entry_count = ref_count = pair_count = 0
    bib_path = output / "bibtex.jsonl"
    refs_path = output / "references.jsonl"
    with bib_path.open("w", encoding="utf-8", newline="\n") as bib_out, refs_path.open("w", encoding="utf-8", newline="\n") as refs_out:
        for index, pid in enumerate(sorted(manual_ids | automatic_ids), 1):
            entries, refs = load_joined_paper(pid, m1_index, m2_index)
            entry_count += len(entries)
            ref_count += len(refs)
            pair_count += len(entries) * len(refs)
            for entry in entries:
                row = {**asdict(entry), "paper_arxiv_id": pid}
                bib_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            for ref in refs:
                row = {**asdict(ref), "arxiv_id": ref.ref_arxiv_id, "paper_arxiv_id": pid, "arxiv_key": ref.ref_arxiv_id}
                refs_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 500 == 0:
                print(f"[{index}/{len(partitions)}] model input papers exported", flush=True)
    sources = sorted(args.entries_dir.glob("r[1-5]_manual_entries.csv")) + [args.automatic_pool, args.automatic_labels]
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "manual_publications": len(manual_ids),
        "manual_labels": len(manual),
        "manual_match_counts": dict(Counter(row["match_status"] for row in manual)),
        "manual_range_counts": dict(Counter(row["range"] for row in pool)),
        "partition_publications": dict(Counter(f"{r['label_source']}/{r['partition']}" for r in partitions)),
        "partition_labels": dict(Counter(f"{r['label_source']}/{r['partition']}" for r in combined)),
        "automatic_integrity_before": integrity,
        "automatic_integrity_after": integrity_after,
        "model_input_bibtex_entries": entry_count,
        "model_input_reference_rows": ref_count,
        "model_input_cartesian_pairs": pair_count,
        "matcher_config": {"threshold": threshold, "minimum_margin": margin},
        "automatic_labels_changed": changes,
        "inputs": [{"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
    }
    _atomic_text(output / "handoff_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(output / "HUONG_DAN_NGOC.md", HANDOFF_GUIDE)
    reviewed_dir = output / "reviewed_csv"
    reviewed_dir.mkdir()
    for source in sorted(args.entries_dir.glob("r[1-5]_manual_entries.csv")):
        shutil.copy2(source, reviewed_dir / source.name)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


HANDOFF_GUIDE = """# Dữ liệu M3 giao Ngọc

## Bắt đầu

Dùng `partitions.jsonl` trong gói này. Range 5 đã được Đức đổi sang
2606.10734, 2606.10742, 2606.10758, 2606.10762, 2606.10791.
Các file split/manual pool ngày trước không còn là bản hiện hành.

Gói chứa 25 paper nhãn tay đã duyệt và 2.498 paper nhãn tự động. Mọi arXiv ID
là STRING; giữ nguyên số 0 đầu/cuối, không đọc qua kiểu float hoặc Excel General.

## File cần cho feature và model

- `bibtex.jsonl`: phẳng, mỗi dòng là một entry. `arxiv_id` và `paper_arxiv_id`
  đều là paper nguồn, `citation_key` xác định entry bên trong paper.
- `references.jsonl`: phẳng, chỉ chứa reference có arXiv ID. `paper_arxiv_id`
  là paper nguồn; `arxiv_id`, `ref_arxiv_id`, `arxiv_key` là ID của REFERENCE.
- `labels.jsonl`: nhãn đã ghép với partition, mỗi dòng một entry đã gán nhãn.
- `labels_train.jsonl`, `labels_valid.jsonl`, `labels_test.jsonl`: nhãn tách sẵn.
- `partitions.jsonl`: chia theo paper, seed 14119; mỗi paper chỉ thuộc một tập.
- `manual_labels.jsonl`, `automatic_labels.jsonl`: nhãn chi tiết để truy nguồn.
- `matcher_calibration.json`: ngưỡng được chọn chỉ bằng manual-train.
- `handoff_manifest.json`: số lượng, cấu hình và checksum nguồn.
- `reviewed_csv/`: bản sao nguyên vẹn 5 CSV Đức đã gửi để truy lại nhãn.

Đọc JSONL phẳng, KHÔNG truyền `list_key` của dạng JSONL lồng cũ:

```python
import pandas as pd
bib = pd.read_json('bibtex.jsonl', lines=True, dtype=False)
refs = pd.read_json('references.jsonl', lines=True, dtype=False)
labels = pd.read_json('labels.jsonl', lines=True, dtype=False)
parts = pd.read_json('partitions.jsonl', lines=True, dtype=False)
```

## Tạo feature và train

1. Ghép theo paper nguồn: `bib.arxiv_id == refs.paper_arxiv_id`.
   Không ghép `refs.arxiv_id` với paper nguồn: đó là ID reference cần dự đoán.
2. Trong cùng paper, tạo cặp entry × candidate reference. Nhãn cặp = 1 nếu
   `ref_arxiv_id == groundtruth_arxiv_id`, còn lại = 0. Ground truth null là
   `no_match`, toàn bộ candidate của entry đó mang nhãn 0.
3. Nếu chỉ một phần entry được gán nhãn, chỉ dùng các entry có mặt trong labels.
4. Fit TF-IDF/scaler/imputer và model bằng Train. Transform Valid/Test bằng
   các đối tượng đã fit. Không fit lại vectorizer trên toàn bộ ba tập.
5. Dùng Validation để chọn feature/model/hyperparameter. Chốt cấu hình rồi
   tính MRR@5 trên Test. Báo riêng manual và automatic vì automatic là silver.
6. Xuất predictions Top 5 theo schema đề. Điểm matcher trong calibration
   không phải điểm MRR@5 của model Ngọc.

## RAM Colab

Dữ liệu trong gói chỉ gồm 2.523 paper cần cho model. Không cần tạo cặp cho
toàn bộ 25.000 paper. Lặp groupby trực tiếp, không gọi `list(groupby)`; ghi
feature theo batch ra Parquet, dùng float32 và giải phóng batch sau khi ghi.
Không giữ toàn bộ ma trận pair của mọi paper trong `parts` rồi concat cuối.
Nếu xuất Parquet theo batch, lưu checkpoint để tiếp tục khi runtime ngắt.

## Giới hạn và nguồn gán nhãn

CSV đã qua kiểm tra đủ số lượng, ID và metadata gốc; kiểm tra cấu trúc không
bảo đảm mọi nhãn ngữ nghĩa đều đúng. Range 4 có trợ giúp AI và được Đức gửi lại
sau bước duyệt. Không mô tả bộ nhãn này là hoàn toàn do người gán độc lập.
Nếu yêu cầu nhãn tay độc lập, thành viên cần xác nhận lại các nhãn có AI hỗ trợ,
đặc biệt phần holdout. Không tự đổi nhãn test theo dự đoán model để tăng điểm.

MRR@5 với entry không có đáp án cần theo quy ước trợ giảng xác nhận. Trong lúc
chưa chốt, xuất đủ predictions và báo cả số entry matched/no_match; có thể lưu
hai biến thể điểm (chỉ matched và toàn bộ với no_match tính 0), ghi rõ cách tính.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries-dir", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m2", type=Path, required=True)
    parser.add_argument("--automatic-pool", type=Path, required=True)
    parser.add_argument("--automatic-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14119)
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
