"""Shared string-ID, split, label and bounded feature-batch contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

from .clean import normalize_arxiv_id


KEY = ["arxiv_id", "citation_key", "ref_arxiv_id"]


def identifier(value):
    if not isinstance(value, str):
        raise ValueError(f"ID must be a string, not {type(value).__name__}: {value!r}")
    canonical = normalize_arxiv_id(value)
    if canonical is None:
        raise ValueError(f"Invalid arXiv ID: {value!r}")
    return canonical


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def handoff_labels(handoff):
    handoff = Path(handoff)
    parts = {}
    for row in read_jsonl(handoff / "partitions.jsonl"):
        pid = identifier(row["arxiv_id"])
        if pid in parts or row["partition"] not in {"train", "valid", "test"}:
            raise ValueError("Duplicate/invalid publication partition")
        parts[pid] = row["partition"]
    rows = read_jsonl(handoff / "labels.jsonl")
    seen = set()
    for row in rows:
        row["arxiv_id"] = identifier(row["arxiv_id"])
        gt = row.get("groundtruth_arxiv_id")
        row["groundtruth_arxiv_id"] = identifier(gt) if gt is not None else None
        key = (row["arxiv_id"], row["citation_key"])
        if key in seen or row.get("partition") != parts.get(row["arxiv_id"]):
            raise ValueError(f"Duplicate label or split mismatch: {key}")
        if row.get("label_source") not in {"manual", "automatic"}:
            raise ValueError(f"Unknown label source: {key}")
        seen.add(key)
    if {row["arxiv_id"] for row in rows} != set(parts):
        raise ValueError("Label/publication coverage mismatch")
    frame = pd.DataFrame(rows)
    # Pandas 3 string inference may replace Python None with NaN. Keep true nulls
    # for JSON/Parquet ground truths and for the no-match evaluation denominator.
    frame["groundtruth_arxiv_id"] = pd.Series([r["groundtruth_arxiv_id"] for r in rows], dtype=object)
    return frame, parts


def merge_labels(pairs, labels):
    """Many candidates join one entry label; null truth means all negatives."""
    pairs = pairs.copy()
    labels = labels.copy()
    for frame in (pairs, labels):
        frame["arxiv_id"] = frame["arxiv_id"].map(identifier)
    pairs["ref_arxiv_id"] = pairs["ref_arxiv_id"].map(identifier)
    labels["groundtruth_arxiv_id"] = labels["groundtruth_arxiv_id"].map(
        lambda value: None if value is None or (isinstance(value, float) and np.isnan(value)) else identifier(value)
    )
    if labels.duplicated(["arxiv_id", "citation_key"]).any():
        raise ValueError("Duplicate entry labels; do not silently drop duplicates")
    keep = ["arxiv_id", "citation_key", "groundtruth_arxiv_id"]
    keep += [name for name in ("partition", "label_source") if name in labels]
    result = pairs.merge(labels[keep], on=["arxiv_id", "citation_key"], how="left", validate="many_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError("Feature entries missing labels after canonical-ID merge")
    result = result.drop(columns="_merge")
    result["label"] = (result["groundtruth_arxiv_id"].notna() & result["ref_arxiv_id"].eq(result["groundtruth_arxiv_id"])).astype("int8")
    return result


def feature_manifest(features):
    path = Path(features) / "features_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("max_cands") is not None:
        raise ValueError("Features must be complete and include all candidates")
    return manifest


def iter_feature_batches(features, labels, partition=None, batch_rows=32768):
    """Read one shard batch at a time and attach labels/partition by entry."""
    features = Path(features)
    selected_labels = labels if partition is None else labels.loc[labels["partition"].eq(partition)]
    selected_ids = pa.array(selected_labels["arxiv_id"].unique())
    for path in sorted(features.glob("batch_*.parquet")):
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=batch_rows):
            batch = batch.filter(pc.is_in(batch.column("arxiv_id"), value_set=selected_ids))
            if not len(batch):
                continue
            frame = batch.to_pandas()
            local_labels = selected_labels.loc[selected_labels["arxiv_id"].isin(frame["arxiv_id"].unique())]
            df = merge_labels(frame, local_labels)
            if not df.empty:
                yield df


def matrix(frame, columns):
    """Deterministic zero imputation; existing missing-indicator features retained."""
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
