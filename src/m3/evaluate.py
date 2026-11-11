"""Entry-level MRR@5, retaining no-match entries in the primary denominator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_data import handoff_labels, identifier, read_jsonl
from .run_m3 import _atomic_text


def reciprocal_rank(truth, candidates):
    if truth is None:
        return 0.0
    for rank, candidate in enumerate(candidates[:5], 1):
        if candidate == truth:
            return 1.0 / rank
    return 0.0


def summarize(rows):
    matched = [row for row in rows if row["groundtruth_arxiv_id"] is not None]
    values = [reciprocal_rank(row["groundtruth_arxiv_id"], [row[f"cand_{i}"] for i in range(1, 6)]) for row in rows]
    total = sum(values)
    return {
        "entries": len(rows), "matched_entries": len(matched), "no_match_entries": len(rows) - len(matched),
        "mrr_at_5": total / len(rows) if rows else None,
        "mrr_at_5_matched_only": total / len(matched) if matched else None,
        "hit_at_1_matched": sum(value == 1 for value in values) / len(matched) if matched else None,
        "hit_at_5_matched": sum(value > 0 for value in values) / len(matched) if matched else None,
    }


def evaluate_predictions(predictions, labels, partition=None):
    if partition is not None:
        labels = labels.loc[labels["partition"].eq(partition)]
    lookup = {(r["arxiv_id"], r["citation_key"]): r for r in labels.to_dict("records")}
    actual = {}
    for row in predictions:
        key = (identifier(row["arxiv_id"]), row["citation_key"])
        if key in actual or key not in lookup:
            raise ValueError(f"Duplicate/unknown prediction entry: {key}")
        label = lookup[key]
        if row["partition"] != label["partition"] or row["groundtruth_arxiv_id"] != label["groundtruth_arxiv_id"]:
            raise ValueError(f"Prediction changes frozen labels/partitions: {key}")
        actual[key] = {**row, "label_source": label["label_source"]}
    if set(actual) != set(lookup):
        raise ValueError("Cannot evaluate a subset silently: prediction entry coverage differs")
    rows = list(actual.values())
    partitions = sorted({r["partition"] for r in rows})
    report = {
        "definition": "Mean reciprocal rank over ALL labelled entries; no_match contributes zero. Matched-only scores are supplementary.",
        "overall": summarize(rows),
        "by_partition": {name: summarize([r for r in rows if r["partition"] == name]) for name in partitions},
        "by_partition_source": {
            name: {source: summarize([r for r in rows if r["partition"] == name and r["label_source"] == source]) for source in ("manual", "automatic")}
            for name in partitions
        },
    }
    return report


def validation_objective(report):
    """Balance manual and automatic validation sources, not their row counts."""
    sources = report["by_partition_source"]["valid"]
    values = [sources[source]["mrr_at_5"] for source in ("manual", "automatic") if sources[source]["entries"]]
    if not values:
        raise ValueError("No Validation entries for model selection")
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels, _ = handoff_labels(args.handoff)
    report = evaluate_predictions(read_jsonl(args.predictions), labels)
    _atomic_text(args.output, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
