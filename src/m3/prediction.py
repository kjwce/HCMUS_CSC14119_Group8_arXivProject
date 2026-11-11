"""Rank all candidates within each entry and export the official nine columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .model_data import feature_manifest, handoff_labels, iter_feature_batches, matrix, sha256
from .parquet_schema import SCHEMAS
from .run_m3 import _atomic_jsonl


def rank_candidates(features, labels, columns, model=None, scaler=None, partition=None):
    top = {}
    for frame in iter_feature_batches(features, labels, partition):
        if model is None:
            scores = frame["combo_score"].to_numpy()
        else:
            values = matrix(frame, columns)
            if scaler is not None:
                values = scaler.transform(values)
            with threadpool_limits(limits=2):
                scores = model.predict_proba(values)[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite candidate scores")
        ranked = frame[["arxiv_id", "citation_key", "ref_arxiv_id"]].copy()
        ranked["score"] = scores
        ranked = ranked.sort_values(["arxiv_id", "citation_key", "score", "ref_arxiv_id"], ascending=[True, True, False, True], kind="stable")
        ranked = ranked.groupby(["arxiv_id", "citation_key"], sort=False).head(5)
        for row in ranked.itertuples(index=False):
            key = (row.arxiv_id, row.citation_key)
            items = top.setdefault(key, [])
            items.append((float(row.score), row.ref_arxiv_id))
        for key in set(zip(ranked["arxiv_id"], ranked["citation_key"])):
            top[key] = sorted(top[key], key=lambda x: (-x[0], x[1]))[:5]
    selected_labels = labels if partition is None else labels.loc[labels["partition"].eq(partition)]
    expected = set(zip(selected_labels["arxiv_id"], selected_labels["citation_key"]))
    if set(top) != expected:
        raise ValueError("Prediction coverage differs from frozen entry labels")
    rows = []
    for label in selected_labels.sort_values(["arxiv_id", "citation_key"]).to_dict("records"):
        candidates = [item[1] for item in top[(label["arxiv_id"], label["citation_key"])]]
        if len(candidates) != len(set(candidates)):
            raise ValueError("Duplicate reference candidates within an entry")
        rows.append({
            "arxiv_id": label["arxiv_id"], "partition": label["partition"],
            "citation_key": label["citation_key"], "groundtruth_arxiv_id": label["groundtruth_arxiv_id"],
            **{f"cand_{i}": candidates[i - 1] if i <= len(candidates) else None for i in range(1, 6)},
        })
    return rows


def write_predictions(output, rows):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists() or (output / "predictions.parquet").exists():
        raise FileExistsError("Predictions already exist")
    _atomic_jsonl(output / "predictions.jsonl", rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS["predictions"]), output / "predictions.parquet", compression="zstd")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = feature_manifest(args.features)
    bundle = joblib.load(args.model)  # Load only model files produced by this project.
    if bundle["features_manifest_sha256"] != sha256(args.features / "features_manifest.json"):
        raise ValueError("Model was trained with a different feature manifest")
    if manifest["input_sha256"]["labels.jsonl"] != sha256(args.handoff / "labels.jsonl"):
        raise ValueError("Frozen labels changed since feature preparation")
    labels, _ = handoff_labels(args.handoff)
    rows = rank_candidates(args.features, labels, bundle["feature_columns"], bundle["model"], bundle["scaler"])
    write_predictions(args.output, rows)
    print(f"Wrote {len(rows)} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
