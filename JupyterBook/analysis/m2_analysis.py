"""
Milestone 2 analysis module for the Jupyter Book (parsing & standardization section).

Same structure as m1_analysis.py: load_all() -> S, tables(S) -> DataFrames, plot_*(S) -> figures,
so the book builds everything at compile time.

Inputs (folder `parsing_stats/`):
  * m2_stats.json (or .jsonl)   - run-level summary written by the parser (one object; if the file
                                  has several lines, the last one is used);
  * paper_stats.jsonl           - one record per publication (also accepted: papers_stats.jsonl;
                                  the file name given in m2_stats["paper_stats_file"] is tried first).

Notes on the numbers
  * Per-paper records are the source of truth for every figure; m2_stats is used for the
    run-level benchmark (wall time, RAM, disk) and cross-checked in tables(S)["consistency"].
  * Versions with no TeX source are recorded as "empty" by the parser. They are counted in the
    coverage statistics and excluded from the deduplication statistics (they contribute 0 elements).
  * "document" is the root node of every parsed version, so it appears once per parsed version in
    the kind and depth statistics.

Usage in the book:
    from m2_analysis import load_all, tables, plot_coverage, ...
    S = load_all("../logs_or_benchmarks/parsing_stats")
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MB, GB = 1024 ** 2, 1024 ** 3
COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
STATS_NAMES = ["m2_stats.json", "m2_stats.jsonl"]
PAPER_NAMES = ["paper_stats.jsonl", "papers_stats.jsonl"]
KIND_ORDER = ["sentence", "formula", "figure", "item", "list", "paragraph",
              "subsubsection", "subsection", "section", "abstract", "document"]
GROUPS = ["1", "2", "3", "4", "5+"]           # number of versions (seen / parsed)
PAPER_CATS = ["all versions parsed", "partially parsed", "no version parsed"]
# The parser (run_m2.py) records these three statuses via store.record_empty_version(), so they are all
# part of m2_stats["empty_versions"]; the coverage chart/table folds them into "empty" by default.
EMPTY_STATUSES = {"missing_directory": "empty", "empty_gathered_text": "empty"}


# ---------------------------------------------------------------- loading
def _read_json(path):
    """A .json file, or a .jsonl file (last non-empty line wins)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [l for l in text.splitlines() if l.strip()]
        return json.loads(lines[-1])


def load_papers(path):
    """Per-paper records -> compact tables (the bulky 'gather' dicts are reduced to counts)."""
    rows, kinds, depths = [], {}, {}
    vstat, bib_src = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            aid = r["arxiv_id"]
            g = list((r.get("gather") or {}).values())
            unres = r.get("unresolved_citation_keys") or {}
            rows.append({
                "arxiv_id": aid,
                "versions_seen": int(r.get("versions_seen", 0)),
                "versions_parsed": int(r.get("versions_parsed", 0)),
                "versions_empty": len(r.get("empty_versions") or []),
                "elements_before": int(r.get("elements_before_dedup", 0)),
                "elements_after": int(r.get("elements_after_dedup", 0)),
                "edges": int(r.get("hierarchy_edges", 0)),
                "max_depth": int(r.get("max_depth", 0)),
                "bib_before": int(r.get("bibtex_entries_before_dedup", 0)),
                "bib_after": int(r.get("bibtex_entries_after_dedup", 0)),
                "unresolved": sum(len(v) for v in unres.values()),
                "used_files": sum(v.get("used_file_count", 0) for v in g),
                "unused_files": sum(v.get("unused_file_count", 0) for v in g),
                "multi_file_versions": sum(1 for v in g if v.get("used_file_count", 0) > 1),
                "missing_includes": sum(len(v.get("missing_files") or []) for v in g),
                "cyclic_includes": sum(len(v.get("cyclic_includes") or []) for v in g),
                "non_utf8_files": sum(len(v.get("non_utf8_files") or []) for v in g),
                "ambiguous_roots": sum(1 for v in g if len(v.get("root_candidates") or []) > 1),
            })
            kinds[aid] = r.get("kind_counts_before_dedup") or {}
            depths[aid] = {int(k): v for k, v in (r.get("depth_counts") or {}).items()}
            for k, st in (r.get("version_status") or {}).items():
                d = vstat.setdefault(int(k), {})
                d[st] = d.get(st, 0) + 1
            for k, v in (r.get("bibtex_source_counts_before_dedup") or {}).items():
                bib_src[k] = bib_src.get(k, 0) + v

    papers = pd.DataFrame(rows).set_index("arxiv_id")
    K = pd.DataFrame.from_dict(kinds, orient="index").reindex(papers.index).fillna(0).astype(int)
    K = K[[k for k in KIND_ORDER if k in K.columns] + sorted(c for c in K.columns if c not in KIND_ORDER)]
    D = pd.DataFrame.from_dict(depths, orient="index").reindex(papers.index).fillna(0).astype(int)
    D = D[sorted(D.columns)]
    return {"papers": papers, "kinds": K, "depths": D, "version_status": vstat, "bib_sources": bib_src}


def load_all(parsing_dir=None):
    here = Path(__file__).resolve().parent
    candidates = [parsing_dir] if parsing_dir else []
    candidates += [
        "parsing_stats",
        "../parsing_stats",
        str(here / "parsing_stats"),
        str(here.parent / "parsing_stats"),
        "../logs_or_benchmarks/parsing_stats",
        "./logs_or_benchmarks/parsing_stats",
    ]
    for c in candidates:
        if not c:
            continue
        d = Path(c)
        sf = next((d / n for n in STATS_NAMES if (d / n).exists()), None)
        if sf is None:
            continue
        summary = _read_json(sf)
        names = [summary.get("paper_stats_file")] + PAPER_NAMES
        pf = next((d / n for n in names if n and (d / n).exists()), None)
        if pf is None:
            continue
        S = load_papers(pf)
        S["summary"] = summary
        return S
    raise FileNotFoundError(f"Could not find m2_stats.json(l) + paper_stats.jsonl in any of: {candidates}")


# ---------------------------------------------------------------- helpers
def _h(n):
    """Compact count label."""
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:,.0f}"


def _grp(s):
    """Number of versions -> '1'..'4', '5+'."""
    return s.map(lambda n: "5+" if n >= 5 else str(int(n)))


def _red(before, after):
    """Reduction in % (NaN when there was nothing to reduce)."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    return np.where(before > 0, 100 * (1 - after / np.where(before > 0, before, 1)), np.nan)


def _paper_categories(p):
    return pd.Series(
        np.where(p.versions_parsed == p.versions_seen, PAPER_CATS[0],
                 np.where(p.versions_parsed == 0, PAPER_CATS[2], PAPER_CATS[1])),
        index=p.index)


def _coverage_by_version(S, merge_empty=True):
    """Version index (v1, v2, ..., v5+) x status (merge_empty=False keeps the parser's raw statuses)."""
    acc = {}
    for idx, d in S["version_status"].items():
        lab = f"v{idx}" if idx < 5 else "v5+"
        for st, n in d.items():
            if merge_empty:
                st = EMPTY_STATUSES.get(st, st)
            acc.setdefault(lab, {})
            acc[lab][st] = acc[lab].get(st, 0) + n
    cov = pd.DataFrame.from_dict(acc, orient="index").fillna(0).astype(int)
    cov = cov.reindex([l for l in ["v1", "v2", "v3", "v4", "v5+"] if l in cov.index])
    first = [c for c in ["parsed", "empty"] if c in cov.columns]
    return cov[first + [c for c in cov.columns if c not in first]]


def _paper_coverage(S):
    p = S["papers"]
    ct = pd.crosstab(_grp(p.versions_seen), _paper_categories(p))
    ct = ct.reindex(index=GROUPS, columns=PAPER_CATS, fill_value=0)
    ct.index.name, ct.columns.name = "Versions seen", None
    return ct


def _dedup_by_group(S):
    """Dedup totals per number of parsed versions (papers with no parsed version excluded)."""
    p = S["papers"]
    p = p[p.versions_parsed > 0]
    g = _grp(p.versions_parsed)
    a = p.groupby(g).agg(**{
        "Papers": ("versions_parsed", "size"), "Versions parsed": ("versions_parsed", "sum"),
        "Elements before": ("elements_before", "sum"), "Elements after": ("elements_after", "sum"),
        "BibTeX before": ("bib_before", "sum"), "BibTeX after": ("bib_after", "sum"),
    }).reindex(GROUPS).fillna(0).astype(int)
    a["Elements reduction %"] = np.round(_red(a["Elements before"], a["Elements after"]), 2)
    a["BibTeX reduction %"] = np.round(_red(a["BibTeX before"], a["BibTeX after"]), 2)
    a.index.name = "Versions parsed"
    return a


def _grouped(ax, cats, series, colors, labels, log=False, fmt=_h):
    n, x = len(series), np.arange(len(cats))
    w = 0.8 / n
    for i, (vals, c, l) in enumerate(zip(series, colors, labels)):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=c, edgecolor="black", alpha=0.85, label=l)
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    if log:
        ax.set_yscale("log")
    ax.margins(y=0.15)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------- tables
def tables(S):
    """Return the DataFrames used in the report."""
    s, p = S["summary"], S["papers"]
    n_p = len(p)
    seen, parsed, empty = int(p.versions_seen.sum()), int(p.versions_parsed.sum()), int(p.versions_empty.sum())
    cats = _paper_categories(p).value_counts()
    eb, ea = int(p.elements_before.sum()), int(p.elements_after.sum())
    bb, ba = int(p.bib_before.sum()), int(p.bib_after.sum())
    unres = int(p.unresolved.sum())
    no_bib = int(((p.bib_before == 0) & (p.unresolved > 0)).sum())
    n_ok = p[p.versions_parsed > 0]

    overview = pd.DataFrame([
        ("Publications processed / failed",
         f"{n_p:,} / {s.get('papers_failed', 0):,}  (parse success rate {100 * s.get('parse_success_rate', 1):.2f}%)"),
        ("Version parse coverage (parsed / requested)", f"{parsed:,} / {seen:,} ({100 * parsed / seen:.2f}%)"),
        ("Empty versions (no TeX source in Milestone 1)", f"{empty:,} ({100 * empty / seen:.2f}%)"),
        ("Papers with all / some / no version parsed",
         f"{cats.get(PAPER_CATS[0], 0):,} / {cats.get(PAPER_CATS[1], 0):,} / {cats.get(PAPER_CATS[2], 0):,}"),
        ("Elements before -> after deduplication", f"{eb:,} -> {ea:,}  ({_red(eb, ea):.2f}% fewer)"),
        ("BibTeX entries before -> after deduplication", f"{bb:,} -> {ba:,}  ({_red(bb, ba):.2f}% fewer)"),
        ("Avg elements per parsed paper (before / after)",
         f"{n_ok.elements_before.mean():,.0f} / {n_ok.elements_after.mean():,.0f}"),
        ("Unresolved citation keys (total / papers with 0 BibTeX entries but unresolved keys)",
         f"{unres:,} / {no_bib:,}"),
        ("Hierarchy depth (mean / max of per-paper max depth)",
         f"{n_ok.max_depth.mean():.2f} / {int(n_ok.max_depth.max())}"),
        ("Versions parsed from multiple files / with unused files ignored",
         f"{int(p.multi_file_versions.sum()):,} / {int((p.unused_files > 0).sum()):,} papers"),
        ("Missing / cyclic includes / non-UTF-8 files",
         f"{int(p.missing_includes.sum()):,} / {int(p.cyclic_includes.sum()):,} / {int(p.non_utf8_files.sum()):,}"),
        ("Versions with several root candidates", f"{int(p.ambiguous_roots.sum()):,}"),
    ], columns=["Metric", "Value"])

    kt = S["kinds"].sum().astype(int)
    kinds = pd.DataFrame({"elements": kt, "share %": (100 * kt / kt.sum()).round(2)})
    kinds.index.name = "kind"

    dt = S["depths"].sum().astype(int)
    depth = pd.DataFrame({"elements": dt, "share %": (100 * dt / dt.sum()).round(2)})
    depth.index.name = "depth"

    cov = _coverage_by_version(S)
    cov["total"] = cov.sum(axis=1)
    if "parsed" in cov.columns:
        cov["parsed %"] = (100 * cov["parsed"] / cov["total"]).round(2)

    chk = [("papers_processed", s.get("papers_processed"), n_p),
           ("versions_requested", s.get("versions_requested"), seen),
           ("versions_parsed", s.get("versions_parsed"), parsed),
           ("empty_versions", s.get("empty_versions"), empty),
           ("elements_before_dedup", s.get("elements_before_dedup"), eb),
           ("elements_after_dedup", s.get("elements_after_dedup"), ea),
           ("bibtex_entries_before_dedup", s.get("bibtex_entries_before_dedup"), bb),
           ("bibtex_entries_after_dedup", s.get("bibtex_entries_after_dedup"), ba),
           ("unresolved_citation_keys", s.get("unresolved_citation_keys"), unres)]
    consistency = pd.DataFrame(
        [(k, a, b, "yes" if a == b else "NO") for k, a, b in chk],
        columns=["Metric", "m2_stats", "Sum of paper_stats", "Match"])

    wall, avg, n_proc = s.get("wall_time_s", 0.0), s.get("avg_seconds_per_processed_paper", 0.0), s.get("papers_processed", n_p)
    out_b = s.get("output_bytes", 0)
    performance = pd.DataFrame([
        ("Benchmark covers the complete corpus", str((s.get("benchmark") or {}).get("complete_corpus_run", "n/a"))),
        ("Workers", f"{s.get('workers', 'n/a')}"),
        ("End-to-end wall time", f"{wall:,.1f} s ({wall / 3600:.2f} h)"),
        ("Throughput", f"{n_proc / wall:.2f} papers/s" if wall else "n/a"),
        ("Wall-clock time per paper (wall time / papers)", f"{wall / n_proc:.3f} s" if n_proc else "n/a"),
        ("Average processing time per paper (reported by the parser)", f"{avg:.3f} s"),
        ("Peak / average RAM", f"{s.get('peak_ram_mb', 0):.1f} MB / {s.get('avg_ram_mb', 0):.1f} MB"),
        ("Final output size", f"{out_b / GB:.2f} GB ({out_b / n_proc / 1024:.1f} KB per paper)" if n_proc else "n/a"),
    ], columns=["Metric", "Value"])

    return {"overview": overview, "coverage": cov, "paper_coverage": _paper_coverage(S),
            "kinds": kinds, "depth": depth, "dedup": _dedup_by_group(S),
            "performance": performance, "consistency": consistency}


# ---------------------------------------------------------------- figures
def _style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def plot_coverage(S):
    """5.1  Grouped bars: parsed vs. empty versions, and paper-level coverage."""
    _style()
    cov, pc = _coverage_by_version(S), _paper_coverage(S)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    ax = axes[0]
    pal = {"parsed": COLORS[2], "empty": COLORS[1]}
    extra = iter(COLORS[3:])
    cols = list(cov.columns)
    colors = [pal.get(c) or next(extra, "#9ca3af") for c in cols]
    _grouped(ax, list(cov.index), [cov[c].values for c in cols], colors, cols, log=True)
    tot, ok = int(cov.values.sum()), int(cov["parsed"].sum()) if "parsed" in cov.columns else 0
    ax.set_title(f"Version-level parsing coverage ({100 * ok / tot:.2f}% parsed)")
    ax.set_xlabel("Version index"), ax.set_ylabel("Versions (log scale)"), ax.legend()

    ax = axes[1]
    _grouped(ax, [f"{g} version{'s' if g != '1' else ''}" for g in pc.index],
             [pc[c].values for c in PAPER_CATS], [COLORS[2], COLORS[3], COLORS[1]], PAPER_CATS, log=True)
    ax.set_title("Paper-level coverage by number of versions")
    ax.set_xlabel("Versions of the paper"), ax.set_ylabel("Papers (log scale)"), ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_kinds(S):
    """5.2  Horizontal bars: elements per kind (before deduplication)."""
    _style()
    kt = S["kinds"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(kt.index, kt.values, color=COLORS[0], edgecolor="black", alpha=0.85)
    for b, v in zip(bars, kt.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,}  ({100 * v / kt.sum():.1f}%)", va="center", fontsize=8)
    ax.set_xscale("log"), ax.set_xlim(right=kt.max() * 12)
    ax.set_xlabel("Elements before deduplication (log scale)")
    ax.set_title("Elements by kind"), ax.grid(axis="x", alpha=0.3)
    return fig


def plot_kind_mix(S):
    """5.2  100% stacked horizontal bars: kind composition by the paper's maximum hierarchy depth."""
    _style()
    p, K = S["papers"], S["kinds"]
    ok = p.elements_before > 0
    lab = p.loc[ok, "max_depth"].map(lambda d: "≤2" if d <= 2 else ("6+" if d >= 6 else str(int(d))))
    order = ["≤2", "3", "4", "5", "6+"]
    mix = K.loc[ok].groupby(lab).sum().reindex(order).dropna(how="all")
    n = lab.value_counts()
    share = mix.div(mix.sum(axis=1), axis=0) * 100
    pal = plt.cm.tab20(np.linspace(0, 1, 20))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for i, k in enumerate(share.columns):
        ax.barh(y, share[k].values, left=left, color=pal[i % 20], edgecolor="white", label=k)
        if k == "sentence":
            for yy, l, v in zip(y, left, share[k].values):
                ax.text(l + v / 2, yy, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += share[k].values
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  (n={n[g]:,})" for g in share.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of elements (%)"), ax.set_ylabel("Max depth of the paper")
    ax.set_title("Element composition by hierarchy depth of the paper")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_depth(S):
    """5.2  Elements per hierarchy level, and papers per maximum depth."""
    _style()
    D = S["depths"].sum()
    p = S["papers"]
    md = p.loc[p.elements_before > 0, "max_depth"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, series, color, xl, yl, title in [
        (axes[0], D, COLORS[0], "Depth (0 = document root)", "Elements (log scale)", "Elements per hierarchy level"),
        (axes[1], md, COLORS[4], "Maximum depth of the paper", "Papers (log scale)", "Papers by maximum hierarchy depth"),
    ]:
        bars = ax.bar(series.index.astype(str), series.values, color=color, edgecolor="black", alpha=0.85)
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _h(b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log"), ax.margins(y=0.15)
        ax.set_xlabel(xl), ax.set_ylabel(yl), ax.set_title(title), ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dedup(S):
    """5.3  Grouped bars: totals before/after deduplication, and reduction by number of versions."""
    _style()
    p = S["papers"]
    eb, ea, bb, ba = (p.elements_before.sum(), p.elements_after.sum(), p.bib_before.sum(), p.bib_after.sum())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    ax = axes[0]
    _grouped(ax, ["Elements", "BibTeX entries"], [[eb / 1e6, bb / 1e6], [ea / 1e6, ba / 1e6]],
             [COLORS[1], COLORS[2]], ["Before deduplication", "After deduplication"], fmt=lambda v: f"{v:.2f}M")
    for x, (b, a) in enumerate([(eb, ea), (bb, ba)]):
        ax.text(x, max(b, a) / 1e6 * 1.12, f"-{_red(b, a):.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Count (millions)"), ax.set_title("Totals before vs. after deduplication"), ax.legend()

    g = _dedup_by_group(S)
    ax = axes[1]
    _grouped(ax, [f"{i}\n(n={int(r['Papers']):,})" for i, r in g.iterrows()],
             [g["Elements reduction %"].fillna(0).values, g["BibTeX reduction %"].fillna(0).values],
             [COLORS[0], COLORS[3]], ["Elements", "BibTeX entries"], fmt=lambda v: f"{v:.1f}%")
    ax.set_xlabel("Versions parsed per paper"), ax.set_ylabel("Reduction (%)")
    ax.set_title("Deduplication gain by number of versions"), ax.legend()
    fig.tight_layout()
    return fig


def plot_performance(S):
    """5.5  Bar charts: time per paper, RAM, output storage (+ tables()['performance'])."""
    _style()
    s, p = S["summary"], S["papers"]
    n = s.get("papers_processed") or len(p)
    wall, avg = s.get("wall_time_s", 0.0), s.get("avg_seconds_per_processed_paper", 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))

    def bars(ax, labels, vals, colors, fmt, ylabel, title):
        bs = ax.bar(labels, vals, color=colors, edgecolor="black", alpha=0.85, width=0.55)
        for b, v in zip(bs, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt(v), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel), ax.set_title(title), ax.margins(y=0.2), ax.grid(axis="y", alpha=0.3)

    bars(axes[0], ["Wall-clock\n(wall / papers)", f"Avg processing\n(reported, {s.get('workers', '?')} workers)"],
         [wall / n, avg], [COLORS[0], COLORS[4]], lambda v: f"{v:.3f} s", "Seconds per paper",
         f"Running time (end-to-end {wall / 3600:.2f} h)")
    bars(axes[1], ["Peak", "Average"], [s.get("peak_ram_mb", 0), s.get("avg_ram_mb", 0)],
         [COLORS[1], COLORS[3]], lambda v: f"{v:.0f} MB", "RAM (MB)", "Memory footprint")
    out = s.get("output_bytes", 0)
    bars(axes[2], ["Final output"], [out / GB], [COLORS[2]], lambda v: f"{v:.2f} GB", "Size (GB)",
         f"Output storage ({out / n / 1024:.0f} KB per paper)")
    fig.tight_layout()
    return fig