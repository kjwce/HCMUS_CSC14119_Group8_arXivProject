"""Command-line preparation tools for Milestone 3.

This module intentionally stops before feature engineering and modelling, which
belong to the teammate responsible for the supervised model.  Generated labels
and partitions are data and must be written outside the Git repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable

from .audit import audit_manual_shortlist
from .clean import normalize_arxiv_id
from .data import index_corpora, inventory_rows, load_joined_paper, read_m2_schema_version
from .matcher import label_publication
from .selection import finalize_manual_pool, select_manual_shortlist
from .split import split_publications
from .validation import validate_automatic_labels


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> int:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(rows)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_label_index(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    seen_rows: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
            if not arxiv_id:
                raise ValueError(f"Missing arxiv_id at {path}:{line_number}")
            citation_key = str(row.get("citation_key") or "").strip()
            if not citation_key:
                raise ValueError(f"Missing citation_key at {path}:{line_number}")
            row_key = (arxiv_id, citation_key)
            if row_key in seen_rows:
                raise ValueError(
                    f"Duplicate label for {arxiv_id}/{citation_key} at {path}:{line_number}"
                )
            seen_rows.add(row_key)
            result.setdefault(arxiv_id, set()).add(citation_key)
    return result


def _read_publication_ids(path: Path) -> set[str]:
    result: set[str] = set()
    for line_number, row in enumerate(_read_jsonl_objects(path), 1):
        arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
        if not arxiv_id:
            raise ValueError(f"Missing arxiv_id at {path}:{line_number}")
        if arxiv_id in result:
            raise ValueError(f"Duplicate publication {arxiv_id} in {path}")
        result.add(arxiv_id)
    return result


def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _range_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH, for example r1=manifest/papers.jsonl")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Range name and manifest path must be non-empty")
    return name.strip(), Path(raw_path.strip())


def _validate_m2_schema(m2_root: Path, required: int) -> int:
    actual = read_m2_schema_version(m2_root)
    if actual is None:
        raise ValueError(
            f"Cannot verify M2 parser schema: {m2_root / 'm2_stats.json'} is missing"
        )
    if actual != required:
        raise ValueError(f"M2 parser schema {actual} found; schema {required} is required")
    return actual


def command_inventory(args: argparse.Namespace) -> int:
    m1_root, m2_root = Path(args.m1), Path(args.m2)
    schema_version = _validate_m2_schema(m2_root, args.required_m2_schema)
    m1_index, m2_index = index_corpora(m1_root, m2_root)
    rows = list(inventory_rows(m1_root, m2_root))
    summary = {
        "m1_publications": len(m1_index),
        "m2_publications": len(m2_index),
        "m2_parser_schema_version": schema_version,
        "joined_publications": len(rows),
        "bibtex_entries": sum(row.bibtex_entries for row in rows),
        "arxiv_references": sum(row.arxiv_references for row in rows),
        "manual_eligible_publications": sum(row.eligible_for_manual_labelling for row in rows),
        "automatic_eligible_publications": sum(
            row.eligible_for_automatic_labelling for row in rows
        ),
    }
    if args.out:
        _atomic_jsonl(Path(args.out), (row.as_dict() for row in rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_auto_label(args: argparse.Namespace) -> int:
    m1_root, m2_root = Path(args.m1), Path(args.m2)
    schema_version = _validate_m2_schema(m2_root, args.required_m2_schema)
    m1_index, m2_index = index_corpora(m1_root, m2_root)
    excluded = _read_publication_ids(Path(args.manual_pool))
    inventory = list(inventory_rows(m1_root, m2_root))
    inventory_by_id = {row.arxiv_id: row for row in inventory}
    eligible = [
        row.arxiv_id
        for row in inventory
        if row.eligible_for_automatic_labelling and row.arxiv_id not in excluded
    ]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if len(eligible) < args.target_papers:
        raise ValueError(
            f"Only {len(eligible)} eligible automatic publications remain; "
            f"requested {args.target_papers}"
        )
    selected = sorted(eligible[: args.target_papers])
    pool_rows = [
        {
            "arxiv_id": arxiv_id,
            "label_source": "automatic",
            "bibtex_entries": inventory_by_id[arxiv_id].bibtex_entries,
            "arxiv_references": inventory_by_id[arxiv_id].arxiv_references,
        }
        for arxiv_id in selected
    ]
    pool_output = Path(args.pool_out)
    _atomic_jsonl(pool_output, pool_rows)

    counters = {"accepted": 0, "no_match": 0, "embedded_arxiv_id": 0}

    def rows() -> Iterable[dict[str, object]]:
        for paper_index, arxiv_id in enumerate(selected, 1):
            entries, references = load_joined_paper(arxiv_id, m1_index, m2_index)
            for row in label_publication(
                entries,
                references,
                threshold=args.threshold,
                minimum_margin=args.minimum_margin,
            ):
                if row["accepted"]:
                    counters["accepted"] += 1
                else:
                    counters["no_match"] += 1
                if row["label_method"] == "embedded_arxiv_id":
                    counters["embedded_arxiv_id"] += 1
                yield row
            if args.progress_every and paper_index % args.progress_every == 0:
                print(
                    f"[{paper_index}/{len(selected)}] automatic publications labelled",
                    flush=True,
                )

    output = Path(args.out)
    label_count = _atomic_jsonl(output, rows())
    summary = {
        "automatic_publications": len(selected),
        "automatic_label_rows": label_count,
        "labels_accepted": counters["accepted"],
        "labels_no_match": counters["no_match"],
        "labels_from_embedded_arxiv_id": counters["embedded_arxiv_id"],
        "m2_parser_schema_version": schema_version,
        "output": str(output.resolve()),
        "pool_output": str(pool_output.resolve()),
        "seed": args.seed,
        "threshold": args.threshold,
        "minimum_margin": args.minimum_margin,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_finalize_manual_pool(args: argparse.Namespace) -> int:
    rows = finalize_manual_pool(
        _read_jsonl_objects(Path(args.audited_shortlist)),
        per_range=args.per_range,
    )
    output = Path(args.out)
    _atomic_jsonl(output, rows)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "ranges": len({str(row["range"]) for row in rows}),
                "manual_publications": len(rows),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_split_pools(args: argparse.Namespace) -> int:
    manual_ids = _read_publication_ids(Path(args.manual_pool))
    automatic_ids = _read_publication_ids(Path(args.automatic_pool))
    rows = split_publications(
        manual_ids,
        automatic_ids,
        holdout_per_source=args.holdout_per_source,
        seed=args.seed,
    )
    output = Path(args.out)
    _atomic_jsonl(output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['label_source']}_{row['partition']}"
        counts[key] = counts.get(key, 0) + 1
    print(
        json.dumps(
            {"output": str(output.resolve()), "publication_counts": counts},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_validate_auto_labels(args: argparse.Namespace) -> int:
    m1_root, m2_root = Path(args.m1), Path(args.m2)
    schema_version = _validate_m2_schema(m2_root, args.required_m2_schema)
    m1_index, m2_index = index_corpora(m1_root, m2_root)
    summary = validate_automatic_labels(
        _read_jsonl_objects(Path(args.automatic_pool)),
        _read_jsonl_objects(Path(args.automatic_labels)),
        m1_index,
        m2_index,
    )
    summary["m2_parser_schema_version"] = schema_version
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_build_manual_package(args: argparse.Namespace) -> int:
    m1_root, m2_root = Path(args.m1), Path(args.m2)
    schema_version = _validate_m2_schema(m2_root, args.required_m2_schema)
    m1_index, m2_index = index_corpora(m1_root, m2_root)
    pool_rows = _read_jsonl_objects(Path(args.manual_pool))
    output_dir = Path(args.output_dir)
    entries_by_range: dict[str, list[dict[str, object]]] = {}
    references_by_range: dict[str, list[dict[str, object]]] = {}
    all_entries: list[dict[str, object]] = []
    all_references: list[dict[str, object]] = []
    for pool_row in pool_rows:
        range_name = str(pool_row.get("range") or "")
        arxiv_id = normalize_arxiv_id(pool_row.get("arxiv_id"))
        if not range_name or not arxiv_id:
            raise ValueError("Manual pool rows require range and arxiv_id")
        entries, references = load_joined_paper(arxiv_id, m1_index, m2_index)
        entry_rows = [
            {
                "range": range_name,
                "arxiv_id": arxiv_id,
                "citation_key": entry.citation_key,
                "bib_title": entry.title or "",
                "bib_authors": " | ".join(entry.authors),
                "bib_year": entry.year if entry.year is not None else "",
                "bib_doi": entry.doi or "",
                "bib_journal": entry.journal or "",
                "bib_url": entry.url or "",
                "match_status": "",
                "groundtruth_arxiv_id": "",
                "annotator_notes": "",
            }
            for entry in entries
        ]
        reference_rows = [
            {
                "range": range_name,
                "arxiv_id": arxiv_id,
                "ref_arxiv_id": reference.ref_arxiv_id,
                "ref_title": reference.title or "",
                "ref_authors": " | ".join(reference.authors),
                "ref_year": reference.year if reference.year is not None else "",
                "ref_submission_date": reference.submission_date or "",
                "ref_s2_id": reference.semantic_scholar_id or "",
            }
            for reference in references
        ]
        entries_by_range.setdefault(range_name, []).extend(entry_rows)
        references_by_range.setdefault(range_name, []).extend(reference_rows)
        all_entries.extend(entry_rows)
        all_references.extend(reference_rows)

    _atomic_csv(output_dir / "manual_entries_all.csv", all_entries)
    _atomic_csv(output_dir / "manual_references_all.csv", all_references)
    for range_name in sorted(entries_by_range):
        _atomic_csv(output_dir / f"{range_name}_manual_entries.csv", entries_by_range[range_name])
        _atomic_csv(
            output_dir / f"{range_name}_manual_references.csv",
            references_by_range[range_name],
        )
    instructions = """# HƯỚNG DẪN GÁN NHÃN THỦ CÔNG — MILESTONE 3

## 1. Mục tiêu

Mỗi dòng trong file `rX_manual_entries.csv` là một mục BibTeX lấy từ phần tài
liệu tham khảo của một paper. Nhiệm vụ là xác định mục BibTeX đó tương ứng với
reference arXiv nào trong file `rX_manual_references.csv` của CÙNG paper.

- BibTeX: thông tin trích dẫn lấy từ LaTeX ở M2, gồm citation key, title,
  authors, year, DOI, journal và URL.
- Reference: metadata Semantic Scholar lấy ở M1. Mỗi reference trong bộ này có
  `ref_arxiv_id` và là một candidate cho BibTeX của cùng publication.
- Ground truth: `ref_arxiv_id` đúng; nếu không có candidate phù hợp thì kết quả
  là `no_match`.

Một BibTeX có tối đa một reference đúng. Tuyệt đối không ghép reference của
paper khác.

## 2. File nào dùng để làm gì?

Mỗi thành viên chỉ dùng cặp file thuộc range của mình:

- `rX_manual_entries.csv`: FILE CẦN ĐIỀN NHÃN và gửi lại.
- `rX_manual_references.csv`: FILE TRA CỨU, không chỉnh sửa.

Hai file `manual_entries_all.csv` và `manual_references_all.csv` là bản tổng hợp
để Đức kiểm tra; thành viên không cần sửa hai file này.

## 3. Số lượng bắt buộc

Mỗi range có 5 paper. Với MỖI paper (`arxiv_id`), phải gán ít nhất 20 BibTeX.
Như vậy mỗi thành viên nộp tối thiểu 5 x 20 = 100 nhãn.

Khuyến khích gán toàn bộ các dòng nếu có thời gian. Nếu chỉ làm đúng 20 dòng,
hãy lấy 20 dòng đầu tiên của từng `arxiv_id` theo thứ tự sẵn có; không được chỉ
chọn các dòng dễ vì sẽ làm tập đánh giá bị thiên lệch.

## 4. Cách làm trong Excel hoặc Google Sheets

1. Mở `rX_manual_entries.csv` và `rX_manual_references.csv`.
2. Lọc cả hai file theo cùng một `arxiv_id`.
3. Chọn một dòng BibTeX trong file entries.
4. Tìm reference phù hợp trong file references bằng title, authors và year.
5. Điền `match_status`, `groundtruth_arxiv_id` và ghi chú nếu cần.
6. Lặp lại cho ít nhất 20 dòng của paper đó rồi chuyển sang paper tiếp theo.

Nên bật Filter và Freeze hàng tiêu đề. Khi lưu bằng Excel, giữ định dạng CSV
UTF-8 để không hỏng ký tự Unicode.

## 5. Quy tắc điền các cột

### Trường hợp tìm được reference đúng

- `match_status`: điền chính xác `matched`
- `groundtruth_arxiv_id`: sao chép chính xác `ref_arxiv_id` từ file references
- `annotator_notes`: có thể để trống hoặc giải thích trường hợp đặc biệt

Ví dụ:

```text
match_status = matched
groundtruth_arxiv_id = 1706.03762
annotator_notes = title và authors trùng; year lệch 1 do bản preprint
```

### Trường hợp không có reference phù hợp

- `match_status`: điền chính xác `no_match`
- `groundtruth_arxiv_id`: để trống
- `annotator_notes`: có thể ghi lý do, ví dụ `không có title tương ứng`

```text
match_status = no_match
groundtruth_arxiv_id =
annotator_notes = không có candidate phù hợp trong paper này
```

### Trường hợp chưa chắc chắn

Không đoán bừa. Tạm để trống hai cột nhãn và ghi `CẦN ĐỨC KIỂM TRA` trong
`annotator_notes`. Dòng chưa chắc chắn chưa được tính vào tối thiểu 20 nhãn cho
đến khi Đức chốt thành `matched` hoặc `no_match`.

## 6. Tiêu chí xác định một match

Ưu tiên bằng chứng theo thứ tự:

1. arXiv ID hoặc DOI xuất hiện trực tiếp và trùng nhau: bằng chứng mạnh nhất.
2. Title giống nhau về nội dung. Bỏ qua khác biệt chữ hoa/thường, dấu câu,
   khoảng trắng, ký hiệu LaTeX hoặc tiêu đề viết tắt nhẹ.
3. Authors trùng phần lớn, đặc biệt tác giả đầu và các họ chính.
4. Year trùng. Có thể lệch 1 năm giữa bản arXiv và bản xuất bản chính thức.
5. Journal/venue hỗ trợ quyết định nhưng không nên dùng một mình.

Không dùng `citation_key` làm bằng chứng chính vì key do tác giả tự đặt. Một
title gần giống nhưng authors khác hoàn toàn thường không phải cùng tài liệu.

## 7. Khi nào chọn no_match?

Chọn `no_match` khi đã xem toàn bộ candidate của cùng `arxiv_id` nhưng:

- không có title tương ứng;
- tài liệu là sách, website hoặc bài không có arXiv;
- candidate gần giống nhưng authors/nội dung là công trình khác;
- Semantic Scholar không trả về tài liệu được BibTeX trích dẫn.

`no_match` là một nhãn hợp lệ, không phải lỗi hay dữ liệu thiếu cần tự bổ sung
từ Internet. Không thêm candidate ngoài file references.

## 8. Các cột tuyệt đối không sửa

Không sửa, xóa hoặc sắp xếp lại nội dung của các cột:

- `range`
- `arxiv_id`
- `citation_key`
- tất cả cột bắt đầu bằng `bib_`

Chỉ điền ba cột:

- `match_status`
- `groundtruth_arxiv_id`
- `annotator_notes`

Không thêm/xóa dòng. Có thể lọc và sắp xếp khi xem, nhưng trước khi gửi phải
giữ lại toàn bộ các dòng ban đầu.

## 9. Checklist trước khi gửi

Với từng `arxiv_id`, kiểm tra:

- [ ] Có ít nhất 20 dòng mang `matched` hoặc `no_match`.
- [ ] Dòng `matched` có `groundtruth_arxiv_id` không rỗng.
- [ ] ID đó tồn tại trong file references của cùng `arxiv_id`.
- [ ] Dòng `no_match` để trống `groundtruth_arxiv_id`.
- [ ] Không còn lỗi chính tả như `match`, `no match`, `Matched`.
- [ ] Không sửa các cột dữ liệu gốc và không xóa dòng.

## 10. File cần nộp lại

Mỗi thành viên chỉ gửi lại file:

```text
rX_manual_entries.csv
```

Không cần gửi lại file references. Đức sẽ gom 5 file entries, kiểm tra tối thiểu
25 paper/500 nhãn và chuyển cho Hưng kiểm tra tự động trước khi giao dữ liệu cho
Ngọc làm model.
"""
    _atomic_text(output_dir / "README_GAN_NHAN.md", instructions)
    _atomic_text(output_dir / "HUONG_DAN_GAN_NHAN.txt", instructions)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "m2_parser_schema_version": schema_version,
                "manual_publications": len(pool_rows),
                "manual_entry_rows": len(all_entries),
                "manual_reference_rows": len(all_references),
                "ranges": len(entries_by_range),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_manual_shortlist(args: argparse.Namespace) -> int:
    inventory_rows_data = _read_jsonl_objects(Path(args.inventory))
    inventory = {
        str(row["arxiv_id"]): row
        for row in inventory_rows_data
        if row.get("arxiv_id")
    }
    manifests: dict[str, list[dict[str, object]]] = {}
    for range_name, path in args.range_manifest:
        if range_name in manifests:
            raise ValueError(f"Duplicate range name: {range_name}")
        manifests[range_name] = _read_jsonl_objects(path)
    rows = select_manual_shortlist(
        inventory,
        manifests,
        per_range=args.per_range,
        seed=args.seed,
        min_bibtex=args.min_bibtex,
        max_bibtex=args.max_bibtex,
        min_references=args.min_references,
        max_references=args.max_references,
    )
    output = Path(args.out)
    _atomic_jsonl(output, rows)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "ranges": len(manifests),
                "publications": len(rows),
                "recommended_publications": sum(bool(row["recommended"]) for row in rows),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_audit_shortlist(args: argparse.Namespace) -> int:
    m1_root, m2_root = Path(args.m1), Path(args.m2)
    schema_version = _validate_m2_schema(m2_root, args.required_m2_schema)
    m1_index, m2_index = index_corpora(m1_root, m2_root)
    rows = audit_manual_shortlist(
        _read_jsonl_objects(Path(args.shortlist)),
        m1_index,
        m2_index,
        final_per_range=args.final_per_range,
    )
    output = Path(args.out)
    _atomic_jsonl(output, rows)
    csv_output = Path(args.csv_out)
    _atomic_csv(csv_output, rows)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "csv_output": str(csv_output.resolve()),
                "m2_parser_schema_version": schema_version,
                "audited_publications": len(rows),
                "final_recommended_publications": sum(
                    bool(row["final_recommended"]) for row in rows
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_split(args: argparse.Namespace) -> int:
    manual_index = _read_label_index(Path(args.manual_labels))
    automatic_index = _read_label_index(Path(args.automatic_labels))
    too_small = {
        arxiv_id: len(keys)
        for arxiv_id, keys in manual_index.items()
        if len(keys) < args.minimum_manual_entries
    }
    if too_small:
        sample = ", ".join(
            f"{arxiv_id} ({count})" for arxiv_id, count in sorted(too_small.items())[:5]
        )
        raise ValueError(
            f"Manual publications need at least {args.minimum_manual_entries} labelled entries; "
            f"too few: {sample}"
        )
    rows = split_publications(
        set(manual_index),
        set(automatic_index),
        holdout_per_source=args.holdout_per_source,
        seed=args.seed,
    )
    output = Path(args.out)
    _atomic_jsonl(output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['label_source']}_{row['partition']}"
        counts[key] = counts.get(key, 0) + 1
    print(
        json.dumps(
            {"output": str(output.resolve()), "publication_counts": counts},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Audit and join the M1/M2 corpora")
    inventory.add_argument("--m1", required=True, help="M1 root or its raw/ directory")
    inventory.add_argument("--m2", required=True, help="M2 processed output root")
    inventory.add_argument("--out", help="Optional per-publication inventory JSONL")
    inventory.add_argument("--required-m2-schema", type=int, default=3)
    inventory.set_defaults(func=command_inventory)

    auto = subparsers.add_parser("auto-label", help="Create classic automatic labels")
    auto.add_argument("--m1", required=True)
    auto.add_argument("--m2", required=True)
    auto.add_argument("--out", required=True)
    auto.add_argument("--pool-out", required=True, help="Frozen automatic publication pool")
    auto.add_argument("--manual-pool", required=True, help="Final manual pool to exclude")
    auto.add_argument("--target-papers", type=int, required=True)
    auto.add_argument("--threshold", type=float, default=0.68)
    auto.add_argument("--minimum-margin", type=float, default=0.05)
    auto.add_argument("--seed", type=int, default=14119)
    auto.add_argument("--progress-every", type=int, default=100)
    auto.add_argument("--required-m2-schema", type=int, default=3)
    auto.set_defaults(func=command_auto_label)

    shortlist = subparsers.add_parser(
        "manual-shortlist", help="Choose manual-label candidates per member range"
    )
    shortlist.add_argument("--inventory", required=True)
    shortlist.add_argument(
        "--range-manifest",
        action="append",
        type=_range_manifest,
        required=True,
        metavar="NAME=PATH",
    )
    shortlist.add_argument("--out", required=True)
    shortlist.add_argument("--per-range", type=int, default=10)
    shortlist.add_argument("--min-bibtex", type=int, default=20)
    shortlist.add_argument("--max-bibtex", type=int, default=60)
    shortlist.add_argument("--min-references", type=int, default=8)
    shortlist.add_argument("--max-references", type=int, default=40)
    shortlist.add_argument("--seed", type=int, default=14119)
    shortlist.set_defaults(func=command_manual_shortlist)

    audit = subparsers.add_parser(
        "audit-shortlist", help="Audit shortlist quality and recommend the final manual pool"
    )
    audit.add_argument("--m1", required=True)
    audit.add_argument("--m2", required=True)
    audit.add_argument("--shortlist", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--csv-out", required=True)
    audit.add_argument("--final-per-range", type=int, default=5)
    audit.add_argument("--required-m2-schema", type=int, default=3)
    audit.set_defaults(func=command_audit_shortlist)

    finalize = subparsers.add_parser(
        "finalize-manual-pool", help="Lock the audited 25-publication manual pool"
    )
    finalize.add_argument("--audited-shortlist", required=True)
    finalize.add_argument("--out", required=True)
    finalize.add_argument("--per-range", type=int, default=5)
    finalize.set_defaults(func=command_finalize_manual_pool)

    pool_split = subparsers.add_parser(
        "split-pools", help="Freeze partitions before feature engineering"
    )
    pool_split.add_argument("--manual-pool", required=True)
    pool_split.add_argument("--automatic-pool", required=True)
    pool_split.add_argument("--out", required=True)
    pool_split.add_argument("--holdout-per-source", type=int, default=5)
    pool_split.add_argument("--seed", type=int, default=14119)
    pool_split.set_defaults(func=command_split_pools)

    validate_auto = subparsers.add_parser(
        "validate-auto-labels", help="Validate automatic-label coverage and candidate scope"
    )
    validate_auto.add_argument("--m1", required=True)
    validate_auto.add_argument("--m2", required=True)
    validate_auto.add_argument("--automatic-pool", required=True)
    validate_auto.add_argument("--automatic-labels", required=True)
    validate_auto.add_argument("--required-m2-schema", type=int, default=3)
    validate_auto.set_defaults(func=command_validate_auto_labels)

    manual_package = subparsers.add_parser(
        "build-manual-package", help="Create human-readable manual-labelling CSV files"
    )
    manual_package.add_argument("--m1", required=True)
    manual_package.add_argument("--m2", required=True)
    manual_package.add_argument("--manual-pool", required=True)
    manual_package.add_argument("--output-dir", required=True)
    manual_package.add_argument("--required-m2-schema", type=int, default=3)
    manual_package.set_defaults(func=command_build_manual_package)

    split = subparsers.add_parser("split", help="Freeze publication-level data partitions")
    split.add_argument("--manual-labels", required=True)
    split.add_argument("--automatic-labels", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--holdout-per-source", type=int, default=5)
    split.add_argument("--minimum-manual-entries", type=int, default=20)
    split.add_argument("--seed", type=int, default=14119)
    split.set_defaults(func=command_split)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
