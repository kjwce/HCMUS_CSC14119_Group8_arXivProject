"""
feature_engineering.py - Feature engineering for Milestone 3 (BibTeX entry <-> reference matching).
Integrated from Ngoc's submitted implementation. The supported final workflow
is ``python -m src.m3.feature_engineering prepare --help``: train-only TF-IDF,
all candidates, canonical string IDs and bounded Parquet output.
Colab-ready: see the cells at the bottom of this docstring.

Problem framing: pairwise (entry, reference) scoring inside ONE publication,
then rank references per entry -> top-5 (MRR@5).

Colab usage
-----------
    !pip install -q rapidfuzz scikit-learn pandas pyarrow tqdm
    from google.colab import drive; drive.mount('/content/drive')
    import sys; sys.path.append('/content/group8-code')  # repository root
    from src.m3 import feature_engineering as fe

    pids = [...]   # ids of the publications you work on (manual + auto pool)
    fe.run_in_batches('/content/drive/MyDrive/m3/bibtex.jsonl', '/content/drive/MyDrive/m3/references.jsonl',
                      pids, train_pids=train_pids, out_dir='/content/features_out', batch_size=100)
    pairs = fe.read_pairs('/content/features_out')  # Small experiments only; final training streams batches.
    labeled = fe.add_labels(pairs, labels_df)   # labels_df: arxiv_id, citation_key, groundtruth_arxiv_id

Column conventions (after loading)
----------------------------------
bib  : arxiv_id (= the PUBLICATION), citation_key, title, authors, year, journal, doi, url
       (a column named paper_arxiv_id is renamed to arxiv_id automatically)
refs : paper_arxiv_id (= the CITING publication), arxiv_key (= the reference, the thing we predict),
       title, authors, year.  If the reference's own id is called arxiv_id and a separate
       paper_arxiv_id column exists, arxiv_id is used as the reference id.
       Edit REF_COLS below if your names differ.

Output
------
build_pairs(bib, refs) -> DataFrame, one row per (arxiv_id, citation_key, ref_arxiv_id) + feature columns.

NOTE: the parsed BibTeX from M2 is noisy:
  * `title` often starts with the co-author surname / year: "Tollaksen, 2011, A new method ..."
  * fields are sometimes shifted: the real title sits in `journal`, the venue sits in `title`
  * authors can contain trailing title text: "P. J. S. G. Ferreira 2014 Direct construction ..."
So we never trust a single field. Besides field-to-field scores, we match the reference
title against a "blob" = title + journal + authors of the entry (containment features).
"""
from __future__ import annotations

import gc
import glob
import json
import os
import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

from .model_data import identifier, merge_labels

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kw):
        return x

# ----- adapt to your references table here ---------------------------------
REF_COLS = dict(
    paper="paper_arxiv_id",   # CITING paper column
    ref_id="arxiv_key",       # reference's own arXiv key (the thing we predict)
    title="title",
    authors="authors",
    year="year",
)
# ---------------------------------------------------------------------------

STOP = set("a an and are as at be by for from in is it of on or the to with via using use".split())
KEY = ["arxiv_id", "citation_key", "ref_arxiv_id"]
BIB_FIELDS = ["citation_key", "title", "authors", "year", "journal", "doi", "url"]


# ============================ 0. loading / prep ============================
def _filter_pids(df: pd.DataFrame, pids: set | None) -> pd.DataFrame:
    if pids is None:
        return df
    cols = [c for c in ("paper_arxiv_id", "arxiv_id") if c in df.columns]
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        mask |= df[c].map(_pid_norm).isin(pids).values
    return df[mask]


def load_table(path: str, list_key: str | None = None, pids=None) -> pd.DataFrame:
    """
    Load a table from Parquet (file, directory, or glob such as data/bibtex/*.parquet)
    or from JSON-lines.  For nested JSONL (one paper per line holding a list of rows under
    `list_key`), the rows are flattened and the parent id is stored in `paper_arxiv_id`.

    pids : optional iterable of publication ids.  Only those publications are kept WHILE READING,
           so the full corpus never has to fit in RAM (this is what prevents the Colab crash).
    """
    pids = {_pid_norm(p) for p in pids} if pids is not None else None

    if path.endswith(".parquet") or os.path.isdir(path) or "*" in path:
        files = sorted(glob.glob(os.path.join(path, "*.parquet"))) if os.path.isdir(path) else sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No parquet files match {path}")
        return pd.concat([_filter_pids(pd.read_parquet(f), pids) for f in files], ignore_index=True)

    if list_key is None:
        return _filter_pids(pd.read_json(path, lines=True), pids)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            paper_id = data.get("paper_arxiv_id") or data.get("arxiv_id")
            if pids is not None and _pid_norm(paper_id) not in pids:
                continue
            for item in data.get(list_key, []) or []:
                item = dict(item)
                item["paper_arxiv_id"] = paper_id
                records.append(item)
    return pd.DataFrame(records)


def _pid_norm(x) -> str:
    """Normalise a publication id for joining: '2011.13456v2' / '2011-13456' -> '2011-13456'."""
    x = str(x).strip().lower()
    x = re.sub(r"v\d+$", "", x)
    return x.replace(".", "-").replace("/", "-")


def prep_bib(bib: pd.DataFrame) -> pd.DataFrame:
    """Make sure every field used downstream exists (the publication id column is chosen in build_pairs)."""
    bib = bib.copy()
    if "arxiv_id" not in bib.columns and "paper_arxiv_id" not in bib.columns:
        raise KeyError(f"bib needs an 'arxiv_id' or 'paper_arxiv_id' column; found {list(bib.columns)}")
    for c in BIB_FIELDS:
        if c not in bib.columns:
            bib[c] = None
    return bib


def prep_refs(refs: pd.DataFrame, paper_id: str | None = None) -> pd.DataFrame:
    """Resolve column names, keep arXiv-keyed references only."""
    c = REF_COLS
    refs = refs.copy()
    if c["ref_id"] not in refs.columns:
        if "arxiv_id" in refs.columns and c["paper"] in refs.columns:
            refs[c["ref_id"]] = refs["arxiv_id"]
        else:
            raise KeyError(f"refs has no '{c['ref_id']}' column; found {list(refs.columns)}. Edit REF_COLS.")
    if c["paper"] not in refs.columns:
        if paper_id is None:
            raise ValueError(f"refs has no '{c['paper']}' column: pass paper_id=...")
        refs[c["paper"]] = paper_id
    for k in ("title", "authors", "year"):
        if c[k] not in refs.columns:
            refs[c[k]] = None
    refs = refs[refs[c["ref_id"]].notna()].copy()
    refs[c["paper"]] = refs[c["paper"]].astype(str)
    return refs


# ============================ 1. text cleaning =============================
def norm(s) -> str:
    """lowercase, strip accents / LaTeX / punctuation, collapse spaces."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\\[a-zA-Z]+", " ", s)          # \emph, \textbf ...
    s = re.sub(r"[{}$\\^_~]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str, drop_stop=True) -> list[str]:
    t = s.split()
    return [w for w in t if w not in STOP] if drop_stop else t


def _as_list(x) -> list:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def last_names(authors) -> list[str]:
    """Last-name tokens. Robust to 'F. Colombo', 'Colombo, F.', and garbled strings."""
    out = []
    for a in _as_list(authors):
        a = str(a)
        if "," in a:
            ln = a.split(",")[0]
        else:
            parts = [p for p in re.split(r"\s+", a.strip()) if p]
            parts = [p for p in parts if not re.fullmatch(r"[A-Za-z]\.?|\d{4}", p)]   # drop initials / years
            ln = parts[-1] if parts else ""
        ln = norm(ln).split()
        if ln:
            out.append(ln[-1])
    return out


def _year(y):
    try:
        y = int(float(y))
        return y if 1800 < y < 2100 else np.nan
    except (TypeError, ValueError):
        return np.nan


_EMB_CACHE: dict[str, np.ndarray] = {}


def _embed(encoder, texts: list[str]) -> np.ndarray:
    """Encode texts once (cache across papers), L2-normalised."""
    new = [t for t in dict.fromkeys(texts) if t not in _EMB_CACHE]
    if new:
        v = np.asarray(encoder.encode(new, show_progress_bar=False, batch_size=64))
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        _EMB_CACHE.update(zip(new, v))
    return np.stack([_EMB_CACHE[t] for t in texts])


# ================== 2. per-publication feature computation =================
def _frac_contained(a_tokens: Iterable[str], b_tokens: set[str]) -> float:
    a = list(a_tokens)
    return sum(t in b_tokens for t in a) / len(a) if a else np.nan


def _rank(M: np.ndarray) -> np.ndarray:
    return (-M).argsort(axis=1).argsort(axis=1) + 1      # 1 = best candidate


def _pair_features(bib_p: pd.DataFrame, ref_p: pd.DataFrame, tfidf: TfidfVectorizer, encoder=None,
                   max_cands: int | None = None) -> pd.DataFrame:
    """All m x n pairs of ONE publication, vectorised with rapidfuzz.cdist."""
    c = REF_COLS
    m, n = len(bib_p), len(ref_p)

    # ---- pre-normalised views ------------------------------------------------
    b_title = [norm(x) for x in bib_p["title"]]
    b_jour = [norm(x) for x in bib_p["journal"]]
    b_auth = [last_names(x) for x in bib_p["authors"]]
    b_auth_txt = [norm(" ".join(map(str, _as_list(x)))) for x in bib_p["authors"]]
    b_blob = [f"{t} {j} {a}".strip() for t, j, a in zip(b_title, b_jour, b_auth_txt)]
    b_year = np.array([_year(y) for y in bib_p["year"]], dtype=float)
    b_ids = [norm(f"{d} {u}") for d, u in zip(bib_p["doi"], bib_p["url"])]
    b_blob_tok = [set(tokens(x)) for x in b_blob]

    r_title = [norm(x) for x in ref_p[c["title"]]]
    r_auth = [last_names(x) for x in ref_p[c["authors"]]]
    r_year = np.array([_year(y) for y in ref_p[c["year"]]], dtype=float)
    r_id = ref_p[c["ref_id"]].astype(str).tolist()
    r_id_norm = [norm(x) for x in r_id]
    r_tok = [tokens(x) for x in r_title]

    # ---- field-to-field fuzzy scores (m x n matrices) ------------------------
    S = {}
    S["title_ratio"] = process.cdist(b_title, r_title, scorer=fuzz.ratio, workers=1)
    S["title_token_set"] = process.cdist(b_title, r_title, scorer=fuzz.token_set_ratio, workers=1)
    S["title_partial"] = process.cdist(b_title, r_title, scorer=fuzz.partial_ratio, workers=1)
    S["journal_vs_title_ratio"] = process.cdist(b_jour, r_title, scorer=fuzz.token_set_ratio, workers=1)
    S["blob_partial"] = process.cdist(b_blob, r_title, scorer=fuzz.partial_ratio, workers=1)
    S["blob_token_set"] = process.cdist(b_blob, r_title, scorer=fuzz.token_set_ratio, workers=1)
    for k in S:
        S[k] = S[k] / 100.0

    # ---- TF-IDF char n-gram cosine (fit on Train only by the final CLI) --------
    A = tfidf.transform(b_blob)
    B = tfidf.transform(r_title)
    S["tfidf_cos"] = (A @ B.T).toarray()
    S["tfidf_cos_title"] = (tfidf.transform(b_title) @ B.T).toarray()

    # ---- token containment / Jaccard / authors / year / id --------------------
    cont = np.full((m, n), np.nan)       # fraction of ref title tokens found in entry blob
    jac = np.zeros((m, n))               # Jaccard of title token sets
    a_hit = np.full((m, n), np.nan)      # fraction of ref last names found in entry blob
    a_jac = np.full((m, n), np.nan)      # Jaccard of last-name sets
    a_first = np.zeros((m, n))           # first ref author is in entry
    a_cnt_diff = np.full((m, n), np.nan)
    year_diff = np.full((m, n), np.nan)
    year_eq = np.full((m, n), np.nan)
    year_in_blob = np.zeros((m, n))
    id_hit = np.zeros((m, n))

    r_tok_set = [set(t) for t in r_tok]
    r_auth_set = [set(a) for a in r_auth]
    for i in range(m):
        bt = set(tokens(b_title[i]))
        bset = b_blob_tok[i]
        ba = set(b_auth[i])
        ba_or_blob = ba | bset
        for j in range(n):
            cont[i, j] = _frac_contained(r_tok[j], bset)
            u = bt | r_tok_set[j]
            jac[i, j] = len(bt & r_tok_set[j]) / len(u) if u else 0.0

            ra = r_auth_set[j]
            if ra:
                a_hit[i, j] = len(ra & ba_or_blob) / len(ra)
                a_first[i, j] = float(r_auth[j][0] in ba_or_blob)
            if ra and ba:
                a_jac[i, j] = len(ra & ba) / len(ra | ba)
            if r_auth[j] and b_auth[i]:
                a_cnt_diff[i, j] = abs(len(r_auth[j]) - len(b_auth[i]))

            if not np.isnan(b_year[i]) and not np.isnan(r_year[j]):
                year_diff[i, j] = abs(b_year[i] - r_year[j])
                year_eq[i, j] = float(year_diff[i, j] == 0)
            if not np.isnan(r_year[j]):
                year_in_blob[i, j] = float(str(int(r_year[j])) in (b_blob[i] + " " + b_ids[i]))

            # arXiv id (e.g. 2103.01234 -> "2103 01234") inside doi / url of the entry
            if r_id_norm[j] and r_id_norm[j] in b_ids[i]:
                id_hit[i, j] = 1.0

    S.update(title_contain=cont, title_jaccard=jac, auth_hit=a_hit, auth_jaccard=a_jac,
             auth_first=a_first, auth_cnt_diff=a_cnt_diff, year_diff=year_diff,
             year_eq=year_eq, year_in_blob=year_in_blob, id_hit=id_hit)

    # ---- length / missing indicators -----------------------------------------
    la = np.array([len(a) for a in b_blob])[:, None]
    lb = np.array([len(b) for b in r_title])[None, :]
    S["len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    S["bib_title_missing"] = np.repeat(np.array([[t == "" for t in b_title]]).T, n, axis=1).astype(float)
    S["bib_auth_missing"] = np.repeat(np.array([[len(a) == 0 for a in b_auth]]).T, n, axis=1).astype(float)
    S["bib_year_missing"] = np.repeat(np.isnan(b_year)[:, None], n, axis=1).astype(float)
    S["ref_auth_missing"] = np.repeat(np.array([[len(a) == 0 for a in r_auth]]), m, axis=0).astype(float)
    S["ref_year_missing"] = np.repeat(np.isnan(r_year)[None, :], m, axis=0).astype(float)
    S["n_refs"] = np.full((m, n), float(n))

    # ---- relative (within-entry) features: ranking depends on competitors ----
    for k in ["title_token_set", "blob_partial", "tfidf_cos", "title_contain", "auth_hit"]:
        M = np.nan_to_num(S[k], nan=0.0)
        S[f"{k}_rank"] = _rank(M)
        S[f"{k}_gap"] = M - M.max(axis=1, keepdims=True)                # 0 for the best
        mu, sd = M.mean(axis=1, keepdims=True), M.std(axis=1, keepdims=True)
        S[f"{k}_z"] = (M - mu) / np.where(sd > 1e-9, sd, 1.0)

    # a single combined score (hand-weighted baseline + useful feature)
    S["combo_score"] = (0.4 * np.nan_to_num(S["blob_partial"]) + 0.3 * np.nan_to_num(S["tfidf_cos"])
                        + 0.2 * np.nan_to_num(S["title_contain"]) + 0.1 * np.nan_to_num(S["auth_hit"]))
    M = S["combo_score"]
    S["combo_score_rank"] = _rank(M)
    S["combo_score_gap"] = M - M.max(axis=1, keepdims=True)

    # ---- OPTIONAL pretrained semantic similarity (graded separately from classic methods) ----
    if encoder is not None:
        sem = _embed(encoder, b_blob) @ _embed(encoder, r_title).T      # unit vectors -> cosine
        S["semantic_cos"] = sem
        S["semantic_cos_rank"] = _rank(sem)
        S["semantic_cos_gap"] = sem - sem.max(axis=1, keepdims=True)

    # ---- flatten to a long DataFrame (optionally keep only the top `max_cands` per entry) ----
    if max_cands and n > max_cands:
        cols = np.argsort(-S["combo_score"], axis=1)[:, :max_cands].ravel()
        rows = np.repeat(np.arange(m), max_cands)
    else:
        rows, cols = np.divmod(np.arange(m * n), n)
    out = pd.DataFrame({
        "arxiv_id": bib_p["arxiv_id"].values[rows],
        "citation_key": bib_p["citation_key"].values[rows],
        "ref_arxiv_id": np.array(r_id)[cols],
    })
    feats = pd.DataFrame({k: v[rows, cols].astype(np.float32) for k, v in S.items()})
    return pd.concat([out, feats], axis=1)


# ============================ 3. public API ================================
def fit_tfidf(bib: pd.DataFrame, refs: pd.DataFrame, max_docs: int = 100_000) -> TfidfVectorizer:
    """Fit char TF-IDF on caller-supplied TRAIN texts; sample only within Train."""
    corpus = [norm(t) for t in refs[REF_COLS["title"]].dropna()]
    corpus += [norm(f"{t} {j}") for t, j in zip(bib["title"], bib["journal"])]
    corpus = [t for t in corpus if t] or ["empty"]
    if len(corpus) > max_docs:
        corpus = list(np.random.RandomState(0).choice(corpus, max_docs, replace=False))
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, dtype=np.float32,
                           min_df=2 if len(corpus) > 50 else 1, max_features=50000).fit(corpus)


def build_pairs(bib: pd.DataFrame, refs: pd.DataFrame, tfidf: TfidfVectorizer | None = None,
                paper_id: str | None = None, encoder=None, entries: pd.DataFrame | None = None,
                max_cands: int | None = None, block: int = 200, target_pids=None,
                out_dir: str | None = None, flush_rows: int = 300_000, verbose: bool = True):
    """
    Candidate pairs of every publication that appears in both tables.

    target_pids: shortcut for `entries`: an iterable of publication ids to process.
    out_dir   : if given, pairs are written to out_dir/batch_XXXX.parquet every `flush_rows` rows and
                the directory path is returned (read back with read_pairs). Otherwise a DataFrame is returned.
    entries   : optional DataFrame restricting the work, with column arxiv_id (publication) and optionally
                citation_key. PASS THIS (your labeled entries / publications): the full corpus does not fit in RAM.
    max_cands : keep only the top-k references per entry by combo_score (None = keep all n).
    block     : entries processed at a time inside a publication (bounds memory to block x n matrices).
    The bibtex publication-id column (arxiv_id or paper_arxiv_id) is chosen automatically as the one
    whose values overlap the references' citing-paper ids.
    """
    bib = prep_bib(bib)
    refs = prep_refs(refs, paper_id)
    refs["_pk"] = refs[REF_COLS["paper"]].map(_pid_norm)

    cands = [c for c in ("arxiv_id", "paper_arxiv_id") if c in bib.columns]
    sample = bib.sample(min(len(bib), 50_000), random_state=0) if len(bib) > 50_000 else bib
    ref_pks = set(refs["_pk"])
    scores = {c: sample[c].map(_pid_norm).isin(ref_pks).mean() for c in cands}
    best = max(scores, key=scores.get)
    if verbose:
        print(f"publication-id overlap with references: {scores} -> using bib['{best}']")
    if scores[best] == 0:
        raise ValueError(
            "No bibtex publication id matches the references' citing-paper ids.\n"
            f"  bib sample ids : {[bib[c].head(3).tolist() for c in cands]}\n"
            f"  refs sample ids: {refs[REF_COLS['paper']].head(3).tolist()}")
    bib["arxiv_id"] = bib[best].astype(str)
    bib["_pk"] = bib["arxiv_id"].map(_pid_norm)

    # ---- restrict BEFORE any heavy work ----
    if entries is None and target_pids is not None:
        entries = pd.DataFrame({"arxiv_id": list(target_pids)})
    if entries is not None:
        ent = entries.copy()
        pid_col = "arxiv_id" if "arxiv_id" in ent.columns else "paper_arxiv_id"
        ent["_pk"] = ent[pid_col].map(_pid_norm)
        if "citation_key" in ent.columns:
            bib = bib.merge(ent[["_pk", "citation_key"]].drop_duplicates(), on=["_pk", "citation_key"], how="inner")
        else:
            bib = bib[bib["_pk"].isin(set(ent["_pk"]))]
    refs = refs[refs["_pk"].isin(set(bib["_pk"]))]
    if verbose:
        print(f"working on {len(bib)} entries, {len(refs)} references, {bib['_pk'].nunique()} publications")

    if tfidf is None:
        raise ValueError("Pass a TF-IDF vectorizer fitted on Train only; use the prepare CLI")
    ref_g = dict(tuple(refs.groupby("_pk")))
    parts, n_rows, shard = [], 0, 0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def flush():
        nonlocal parts, n_rows, shard
        if out_dir and parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                os.path.join(out_dir, f"batch_{shard:04d}.parquet"), index=False)
            parts, n_rows, shard = [], 0, shard + 1
            gc.collect()

    for pk, bib_p in tqdm(bib.groupby("_pk", sort=False), total=bib["_pk"].nunique(), desc="publications", disable=not verbose):
        if pk not in ref_g:
            continue
        ref_p = ref_g[pk].reset_index(drop=True)
        bib_p = bib_p.reset_index(drop=True)
        for start in range(0, len(bib_p), block):         # features are within-entry -> row blocks are safe
            part = _pair_features(bib_p.iloc[start:start + block].reset_index(drop=True),
                                  ref_p, tfidf, encoder, max_cands)
            parts.append(part)
            n_rows += len(part)
            if out_dir and n_rows >= flush_rows:
                flush()
    if out_dir:
        flush()
        print(f"features written to {out_dir}/ ({shard} shards)")
        return out_dir
    if not parts:
        return pd.DataFrame(columns=KEY)
    return pd.concat(parts, ignore_index=True)


def read_pairs(out_dir: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read back the shards written by build_pairs(..., out_dir=...)."""
    files = sorted(glob.glob(os.path.join(out_dir, "batch_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no shards in {out_dir}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def add_labels(pairs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    labels: columns arxiv_id, citation_key, groundtruth_arxiv_id (NaN = no true reference).
    Keeps ONLY labeled entries; an entry with NaN groundtruth gets an all-zero row.
    """
    return merge_labels(pairs, labels)


NON_FEATURE = set(KEY) | {"groundtruth_arxiv_id", "label", "partition", "label_source", "match_status"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE]


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """LGBMRanker `group=`: number of candidates per entry (df must be sorted by KEY[:2])."""
    return df.groupby(KEY[:2], sort=False).size().values


# ====================== batch runner (bounded RAM, resumable) ======================
def index_jsonl(path: str) -> dict[str, int]:
    """One cheap pass over a nested JSONL: normalised publication id -> byte offset of its line."""
    offsets, pos = {}, 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pid = d.get("paper_arxiv_id") or d.get("arxiv_id")
                offsets[_pid_norm(pid)] = pos
            pos += len(line)
    return offsets


def load_by_offsets(path: str, list_key: str, offsets: dict[str, int], pids) -> pd.DataFrame:
    """Read ONLY the lines of `pids` (random access), flattened like load_table."""
    records = []
    with open(path, "rb") as f:
        for p in pids:
            f.seek(offsets[p])
            d = json.loads(f.readline())
            paper_id = d.get("paper_arxiv_id") or d.get("arxiv_id")
            for item in d.get(list_key, []) or []:
                item = dict(item)
                item["paper_arxiv_id"] = paper_id
                records.append(item)
    return pd.DataFrame(records)


def run_in_batches(bib_path: str, refs_path: str, pids, out_dir: str | None = None, batch_size: int = 100,
                   bib_key: str = "bibtex_entries", refs_key: str = "references",
                   max_cands: int | None = None, block: int = 100, tfidf_pubs: int = 300,
                   train_pids=None) -> str:
    """
    Feature engineering for `pids`, `batch_size` publications at a time.
      * only the current batch is ever in RAM (random access into the JSONL files),
      * one TF-IDF model is fitted once and reused (features comparable across batches),
      * out_dir defaults to the folder that holds bib_path (the very same path as the inputs);
      * one shard per batch: out_dir/batch_XXXX.parquet; shards that already exist are skipped,
        so after a crash / disconnect you just run the same call again.
    Read the result with read_pairs(out_dir).
    """
    import pickle
    if train_pids is None:
        raise ValueError("train_pids is required for TF-IDF fitting; use the prepare CLI")
    if out_dir is None:                      # default: the SAME folder as the input files
        out_dir = os.path.dirname(os.path.abspath(bib_path))
    os.makedirs(out_dir, exist_ok=True)
    print("indexing files (one pass each)...")
    ib, ir = index_jsonl(bib_path), index_jsonl(refs_path)
    norm_pids = list(dict.fromkeys(_pid_norm(p) for p in pids))
    ok = [p for p in norm_pids if p in ib and p in ir]
    print(f"{len(ok)}/{len(norm_pids)} publications found in both files")
    if not ok:
        raise ValueError("none of the pids appear in both files")

    tf_path = os.path.join(out_dir, "tfidf.pkl")
    if os.path.exists(tf_path):
        raise ValueError("Legacy cached TF-IDF has no split provenance; use a new output directory")
    else:
        train_set = {_pid_norm(pid) for pid in train_pids}
        eligible = [pid for pid in ok if pid in train_set]
        if not eligible:
            raise ValueError("No training publications for TF-IDF")
        samp = list(np.random.RandomState(0).permutation(eligible)[:tfidf_pubs])
        tfidf = fit_tfidf(prep_bib(load_by_offsets(bib_path, bib_key, ib, samp)),
                          prep_refs(load_by_offsets(refs_path, refs_key, ir, samp)))
        pickle.dump(tfidf, open(tf_path, "wb"))

    n_batches = (len(ok) + batch_size - 1) // batch_size
    for bi in tqdm(range(n_batches), desc="batches"):
        shard = os.path.join(out_dir, f"batch_{bi:04d}.parquet")
        if os.path.exists(shard):
            continue
        chunk = ok[bi * batch_size:(bi + 1) * batch_size]
        bib = load_by_offsets(bib_path, bib_key, ib, chunk)
        refs = load_by_offsets(refs_path, refs_key, ir, chunk)
        pairs = build_pairs(bib, refs, tfidf=tfidf, max_cands=max_cands, block=block, verbose=False)
        pairs.to_parquet(shard + ".tmp", index=False)
        os.replace(shard + ".tmp", shard)          # a shard is either complete or absent
        del bib, refs, pairs
        gc.collect()
    print(f"done -> {out_dir}  (load with read_pairs)")
    return out_dir


# ============================== quick demo =================================
def run_demo(bib_path: str, refs_path: str, target_pids=None, out_dir: str = "features_out",
             bib_key="bibtex_entries", refs_key="references", paper_id: str | None = None):
    print("1. Loading BibTeX data...")
    bib = load_table(bib_path, bib_key, pids=target_pids)
    print(f"   -> {len(bib)} BibTeX entries")
    print("2. Loading References data...")
    refs = load_table(refs_path, refs_key, pids=target_pids)
    print(f"   -> {len(refs)} references")
    print("3. Building features (written to disk)...")
    build_pairs(bib, refs, paper_id=paper_id, out_dir=out_dir)
    P = read_pairs(out_dir, columns=KEY + ["combo_score", "title_contain"])
    print(f"4. Done. {len(P)} candidate pairs")
    print(P.sort_values("combo_score", ascending=False).head(8))
    return out_dir


def feature_descriptions(names):
    base = {
        "title_ratio": "Character edit similarity between entry and reference titles.",
        "title_token_set": "Title token-set similarity tolerates reordered or extra words.",
        "title_partial": "Partial title similarity handles shortened titles.",
        "journal_vs_title_ratio": "Detects reference titles stored in the entry journal field.",
        "blob_partial": "Partial reference-title similarity against title, journal and authors.",
        "blob_token_set": "Token-set similarity against the combined entry text.",
        "tfidf_cos": "Train-fitted character n-gram TF-IDF cosine against combined entry text.",
        "tfidf_cos_title": "Train-fitted character n-gram TF-IDF title-only cosine.",
        "title_contain": "Fraction of reference-title tokens appearing in the combined entry text.",
        "title_jaccard": "Jaccard overlap between title token sets.",
        "auth_hit": "Fraction of reference author surnames in entry authors or combined text.",
        "auth_jaccard": "Jaccard overlap of author surname sets.",
        "auth_first": "Whether the reference first author surname occurs in the entry.",
        "auth_cnt_diff": "Absolute difference in author counts; truncated metadata can change it.",
        "year_diff": "Absolute publication-year difference, allowing the model to handle preprints.",
        "year_eq": "Whether publication years agree exactly.",
        "year_in_blob": "Whether the reference year occurs in combined entry text or identifiers.",
        "id_hit": "Whether the normalized reference arXiv identifier appears in entry DOI/URL.",
        "len_ratio": "Shorter/longer combined-entry and reference-title character length ratio.",
        "bib_title_missing": "Entry title is missing.",
        "bib_auth_missing": "Entry author list is missing.",
        "bib_year_missing": "Entry year is missing or invalid.",
        "ref_auth_missing": "Reference author list is missing.",
        "ref_year_missing": "Reference year is missing or invalid.",
        "n_refs": "Number of candidate references within the publication.",
        "combo_score": "Fixed baseline: .4 blob_partial + .3 tfidf_cos + .2 title_contain + .1 auth_hit.",
    }
    result = {}
    for name in names:
        if name in base:
            result[name] = base[name]
        else:
            parent, suffix = name.rsplit("_", 1)
            result[name] = {
                "rank": "Descending within-entry rank (1 is best). ",
                "gap": "Within-entry score minus the maximum (0 is best). ",
                "z": "Within-entry standardized score, using all reference candidates. ",
            }[suffix] + base[parent]
    return result


def prepare_features(handoff, output, block=100, shard_rows=250000, max_docs=100000):
    """Rebuild Ngoc's 43 features on the frozen pool with train-only vocabulary."""
    import hashlib
    import time
    from pathlib import Path
    import joblib
    import pyarrow as pa
    import pyarrow.parquet as pq
    import psutil
    from .model_data import handoff_labels, read_jsonl, sha256
    from .run_m3 import _atomic_text

    handoff, output = Path(handoff), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new feature output directory: {output}")
    if min(block, shard_rows, max_docs) < 1:
        raise ValueError("Block, shard and document limits must be positive")
    started = time.perf_counter()
    labels, parts = handoff_labels(handoff)
    bib = pd.DataFrame(read_jsonl(handoff / "bibtex.jsonl"))
    refs = pd.DataFrame(read_jsonl(handoff / "references.jsonl"))
    bib["arxiv_id"] = bib["arxiv_id"].map(identifier)
    refs["paper_arxiv_id"] = refs["paper_arxiv_id"].map(identifier)
    refs["arxiv_key"] = refs["ref_arxiv_id"].map(identifier)
    if set(bib["arxiv_id"]) != set(parts) or set(refs["paper_arxiv_id"]) != set(parts):
        raise ValueError("Input paper coverage differs from frozen partitions")
    if bib.duplicated(["arxiv_id", "citation_key"]).any() or refs.duplicated(["paper_arxiv_id", "arxiv_key"]).any():
        raise ValueError("Duplicate entry/candidate IDs in handoff")
    bib_keys = set(zip(bib["arxiv_id"], bib["citation_key"]))
    if bib_keys != set(zip(labels["arxiv_id"], labels["citation_key"])):
        raise ValueError("Feature entry coverage must equal reviewed/automatic labels")
    train_ids = {pid for pid, split in parts.items() if split == "train"}
    print(f"Fitting TF-IDF from {len(train_ids)} TRAIN publications only...", flush=True)
    tfidf = fit_tfidf(bib[bib["arxiv_id"].isin(train_ids)], refs[refs["paper_arxiv_id"].isin(train_ids)], max_docs=max_docs)
    output.mkdir(parents=True)
    joblib.dump(tfidf, output / "tfidf.joblib")
    _atomic_text(output / "_IN_PROGRESS", "Feature preparation not complete.\n")
    ref_groups = refs.groupby("paper_arxiv_id", sort=False).indices
    bib_groups = bib.groupby("arxiv_id", sort=False).indices
    schema = None
    writer = None
    names = []
    shard_paths = []
    shard_count = 0
    written = expected = 0
    peak = psutil.Process().memory_info().rss
    try:
        for number, pid in enumerate(sorted(parts), 1):
            b = bib.iloc[bib_groups[pid]].reset_index(drop=True)
            r = refs.iloc[ref_groups[pid]].reset_index(drop=True)
            if r.empty:
                raise ValueError(f"Matching pool has no candidates: {pid}")
            expected += len(b) * len(r)
            for offset in range(0, len(b), block):
                frame = _pair_features(b.iloc[offset:offset + block].reset_index(drop=True), r, tfidf, max_cands=None)
                if schema is None:
                    names = feature_columns(frame)
                    schema = pa.schema([pa.field(key, pa.string(), nullable=False) for key in KEY] + [pa.field(name, pa.float32()) for name in names])
                table = pa.Table.from_pandas(frame[KEY + names], preserve_index=False).cast(schema)
                if writer is None:
                    shard_path = output / f"batch_{len(shard_paths):04d}.parquet"
                    writer = pq.ParquetWriter(shard_path.with_suffix(".tmp"), schema, compression="zstd")
                    shard_count = 0
                writer.write_table(table)
                written += len(table)
                shard_count += len(table)
                if shard_count >= shard_rows:
                    writer.close()
                    shard_path.with_suffix(".tmp").replace(shard_path)
                    shard_paths.append(shard_path)
                    writer = None
            peak = max(peak, psutil.Process().memory_info().rss)
            if number % 100 == 0 or number == len(parts):
                print(f"[{number}/{len(parts)}] publications; {written:,} full candidate pairs", flush=True)
        if writer is not None:
            writer.close()
            shard_path.with_suffix(".tmp").replace(shard_path)
            shard_paths.append(shard_path)
            writer = None
    finally:
        if writer is not None:
            writer.close()
    if written != expected or sum(pq.ParquetFile(path).metadata.num_rows for path in shard_paths) != expected:
        raise ValueError("Full candidate matrix row count mismatch")
    manifest = {
        "status": "complete", "recipe": "Ngoc's classic 43-feature implementation, integrated",
        "max_cands": None, "pair_rows": written, "publications": len(parts), "entries": len(bib),
        "feature_columns": names, "feature_descriptions": feature_descriptions(names),
        "tfidf_fit_partition": "train", "tfidf_allowed_training_publications": sorted(train_ids),
        "tfidf_max_documents": max_docs, "pretrained_encoder": None,
        "shards": len(shard_paths), "wall_time_s": round(time.perf_counter() - started, 3),
        "peak_sampled_rss_mib": round(peak / 1024**2, 3),
        "input_sha256": {name: sha256(handoff / name) for name in ("bibtex.jsonl", "references.jsonl", "labels.jsonl", "partitions.jsonl")},
        "tfidf_sha256": sha256(output / "tfidf.joblib"),
    }
    _atomic_text(output / "features_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "_IN_PROGRESS").unlink()
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"feature_descriptions", "tfidf_allowed_training_publications"}}, indent=2))
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare final full-candidate features from the frozen handoff")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--handoff", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--block", type=int, default=100)
    prepare.add_argument("--shard-rows", type=int, default=250000)
    prepare.add_argument("--max-docs", type=int, default=100000)
    args = parser.parse_args()
    prepare_features(args.handoff, args.output, args.block, args.shard_rows, args.max_docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
