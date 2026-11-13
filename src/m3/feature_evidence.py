"""Evidence for the README feature rationale and result analysis.

Recomputes the 43 model features with ``feature_engineering.build_pairs`` on

* every reviewed manual publication (human labels), and
* a seeded random sample of automatic-label Train publications (silver labels),

then reports, per feature, the match/nonmatch medians, single-feature ROC-AUC
and how often the true reference is the unique best candidate. It also
classifies reviewed ``no_match`` entries and computes the MRR@5 ceiling of each
partition from the published ``predictions`` table.

Only published/frozen inputs are read; nothing is retrained or modified::

    python -m src.m3.feature_evidence `
      --dataset "D:\\m3_work\\hf_dataset_v1" `
      --manual "D:\\m3_work\\manual_csv" `
      --output "D:\\m3_work\\feature_evidence.json"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import feature_engineering as fe

SEED = 14119
TOP1_FEATURES = [
    "title_ratio", "title_token_set", "title_partial", "title_jaccard", "blob_partial", "tfidf_cos",
    "title_contain", "auth_hit", "auth_jaccard", "auth_first", "year_eq", "id_hit", "combo_score",
]
# arXiv identifiers written in a BibTeX URL/DOI/journal/title field.
ARXIV_IN_TEXT = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.|arxiv[:\s]+)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})", re.I
)


def _id(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = str(value).strip()
    return value or None


def load_manual(folder: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "r*_manual_entries.csv"))):
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows += list(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No r*_manual_entries.csv files in {folder}")
    manual = pd.DataFrame(rows, dtype=str)
    manual["gt"] = manual["groundtruth_arxiv_id"].map(_id)
    manual.loc[manual["match_status"] == "no_match", "gt"] = None
    return manual


def feature_stats(pairs: pd.DataFrame, features: list[str]) -> dict:
    out = {}
    for name in features:
        x = pd.to_numeric(pairs[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ok = x.notna()
        y = pairs.loc[ok, "label"]
        auc = None
        if y.nunique() == 2 and x[ok].nunique() > 1:
            auc = round(float(roc_auc_score(y, x[ok])), 4)
        match, nonmatch = x[pairs["label"] == 1].dropna(), x[pairs["label"] == 0].dropna()
        out[name] = {
            "match_median": None if match.empty else round(float(match.median()), 3),
            "nonmatch_median": None if nonmatch.empty else round(float(nonmatch.median()), 3),
            "auc": auc,
        }
    return out


def unique_top1(pairs: pd.DataFrame, name: str) -> dict:
    """Share of matched entries whose true reference has the strictly highest value."""
    hits = total = 0
    for _, entry in pairs[pairs["gt"].notna()].groupby(["arxiv_id", "citation_key"]):
        if entry["label"].sum() != 1:
            continue
        total += 1
        x = pd.to_numeric(entry[name], errors="coerce").fillna(-np.inf)
        best = x.max()
        hits += int(x[entry["label"] == 1].iloc[0] == best and int((x == best).sum()) == 1)
    return {"share": round(hits / total, 4) if total else None, "entries": total}


def classify_no_match(manual: pd.DataFrame, refs: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    candidates = refs.groupby("paper_arxiv_id")["arxiv_key"].apply(lambda s: set(map(str, s))).to_dict()
    best = pairs[pairs["source"] == "manual"].groupby(["arxiv_id", "citation_key"])["title_token_set"].max()
    rows = []
    for row in manual.itertuples(index=False):
        text = " ".join(str(getattr(row, f) or "") for f in ("bib_url", "bib_doi", "bib_journal", "bib_title"))
        ids = {m.group(1).lower() for m in ARXIV_IN_TEXT.finditer(text)}
        rows.append({
            "range": row.range, "arxiv_id": row.arxiv_id, "citation_key": row.citation_key,
            "status": row.match_status, "has_title": bool(str(row.bib_title or "").strip()),
            "arxiv_in_bib": bool(ids), "arxiv_in_candidates": bool(ids & candidates.get(row.arxiv_id, set())),
            "best_title_token_set": best.get((row.arxiv_id, row.citation_key)), "ids": sorted(ids),
        })
    entries = pd.DataFrame(rows)
    no, matched = entries[entries["status"] == "no_match"], entries[entries["status"] == "matched"]
    reason = np.select(
        [~no["has_title"], no["arxiv_in_bib"] & ~no["arxiv_in_candidates"], no["arxiv_in_bib"]],
        ["bib_broken", "s2_missing", "check_label"], default="no_arxiv_evidence")
    return {
        "entries": int(len(entries)), "no_match": int(len(no)), "matched": int(len(matched)),
        "by_range": {r: {"entries": int(len(g)), "no_match": int((g["status"] == "no_match").sum())}
                     for r, g in entries.groupby("range")},
        "reasons": {k: int(v) for k, v in Counter(reason).items()},
        "s2_missing_examples": no[no["arxiv_in_bib"] & ~no["arxiv_in_candidates"]][
            ["arxiv_id", "citation_key", "ids"]].to_dict("records"),
        "matched_with_arxiv_in_bib": int(matched["arxiv_in_bib"].sum()),
        "best_title_token_set": {
            "matched_median": round(float(matched["best_title_token_set"].median()), 3),
            "no_match_median": round(float(no["best_title_token_set"].median()), 3),
            "no_match_at_least_0.9": int((no["best_title_token_set"] >= 0.9).sum()),
            "no_match_below_0.6": int((no["best_title_token_set"] < 0.6).sum()),
            "matched_below_0.6": int((matched["best_title_token_set"] < 0.6).sum()),
        },
    }


def mrr_ceiling(pred: pd.DataFrame) -> dict:
    """All-entry MRR@5 cannot exceed the share of entries that have a ground truth."""
    share = lambda g: round(float(g["groundtruth_arxiv_id"].notna().mean()), 4) if len(g) else None
    return {p: {"entries": int(len(g)), "ceiling": share(g), "manual": share(g[g["is_manual"]]),
                "automatic": share(g[~g["is_manual"]])} for p, g in pred.groupby("partition")}


def run(args: argparse.Namespace) -> dict:
    table = lambda name: os.path.join(args.dataset, "data", name)
    manual = load_manual(args.manual)
    manual_pids = set(manual["arxiv_id"])
    pred = pd.read_parquet(table("predictions"))
    pred["partition"] = pred["partition"].astype(str)
    pred["is_manual"] = pred["arxiv_id"].isin(manual_pids)
    partition = dict(zip(pred["arxiv_id"], pred["partition"]))

    silver_train = sorted(set(pred.loc[~pred["is_manual"] & (pred["partition"] == "train"), "arxiv_id"]))
    rng = np.random.RandomState(SEED)
    silver = set(rng.choice(silver_train, min(args.silver_papers, len(silver_train)), replace=False))
    pids = manual_pids | silver

    bib = fe.load_table(table("bibtex"), pids=pids)
    refs = fe.load_table(table("references"), pids=pids).rename(columns={
        "arxiv_id": "paper_arxiv_id", "ref_arxiv_id": "arxiv_key", "ref_title": "title", "ref_authors": "authors"})
    refs["year"] = refs["ref_submission_date"].astype(str).str[:4]
    train = {p for p in pids if partition.get(p) == "train"}
    tfidf = fe.fit_tfidf(bib[bib["arxiv_id"].isin(train)], refs[refs["paper_arxiv_id"].isin(train)])
    pairs = fe.build_pairs(bib, refs, tfidf=tfidf, target_pids=pids, verbose=False)

    labels = pd.concat([
        manual[["arxiv_id", "citation_key", "gt"]].assign(source="manual"),
        pred.loc[pred["arxiv_id"].isin(silver), ["arxiv_id", "citation_key", "groundtruth_arxiv_id"]]
            .rename(columns={"groundtruth_arxiv_id": "gt"}).assign(source="automatic"),
    ])
    pairs = pairs.merge(labels, on=["arxiv_id", "citation_key"], how="inner")
    pairs["label"] = (pairs["ref_arxiv_id"].astype(str) == pairs["gt"].astype(str)).astype(int)
    features = [c for c in fe.feature_columns(pairs) if c not in ("gt", "source")]

    sources = ("manual", "automatic")
    result = {
        "seed": SEED,
        "sample": {"manual_publications": len(manual_pids), "automatic_publications": len(silver),
                   "pairs": {s: int((pairs["source"] == s).sum()) for s in sources},
                   "positive_pairs": {s: int(pairs.loc[pairs["source"] == s, "label"].sum()) for s in sources}},
        "feature_stats": {s: feature_stats(pairs[pairs["source"] == s], features) for s in sources},
        "unique_top1": {s: {f: unique_top1(pairs[pairs["source"] == s], f) for f in TOP1_FEATURES if f in pairs}
                        for s in sources},
        "no_match": classify_no_match(manual, refs, pairs),
        "mrr_ceiling": mrr_ceiling(pred),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Final six-table dataset folder (README.md + data/)")
    parser.add_argument("--manual", required=True, help="Folder with the reviewed r*_manual_entries.csv files")
    parser.add_argument("--output", required=True, help="JSON report to write")
    parser.add_argument("--silver-papers", type=int, default=150,
                        help="Automatic-label Train publications sampled for comparison")
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("sample", "mrr_ceiling")}, indent=2))
    print(json.dumps({k: v for k, v in result["no_match"].items() if k != "s2_missing_examples"}, indent=2))


if __name__ == "__main__":
    main()
