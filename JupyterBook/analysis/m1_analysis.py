"""
Milestone 1 analysis module for the Jupyter Book (results & performance section).

Derived from cell 3 of test_scripts/Milestone1_scraper.ipynb, with these changes:
  * works on a folder of per-range logs (range_logs/r1..r5), no Google Drive needed;
  * keeps per-range statistics next to the pooled ones;
  * size and latency statistics only use versions whose measurements really exist.
    Range 2 lost part of its log, and 5,206 of its 'ok' versions were later re-registered
    from files already on disk (note = "recovered_existing_files"), without byte counts or
    timings. Counting those as 0 would bias the "before figure removal" average and the
    storage reduction;
  * returns DataFrames and matplotlib figures so the book builds everything at compile time.

Usage in the book:
    from m1_analysis import load_all, tables, plot_versions, ...
    S = load_all("../logs_or_benchmarks/range_logs")
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FINAL = {"ok", "bib_only", "pdf_only", "non_tex", "no_source"}
RECOVERED = "recovered_existing_files"
MB, GB = 1024 ** 2, 1024 ** 3
COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]


def _f(x):
    """CSV cell -> float, '' -> None."""
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


def _csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_range(d):
    """All statistics of one range folder (expects manifest/ and logs/)."""
    d = Path(d)
    papers = {}
    for line in open(d / "manifest" / "papers.jsonl", encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            papers[rec["arxiv_id"]] = rec
    dl = {(r["arxiv_id"], r["version"]): r for r in _csv(d / "logs" / "download_log.csv")}
    s2 = {r["arxiv_id"]: r for r in _csv(d / "logs" / "s2_log.csv")}
    res = _csv(d / "logs" / "resource_log.csv")

    runs = []
    for f in sorted(glob.glob(str(d / "logs" / "run_summary_*.json"))):
        rep = json.loads(Path(f).read_text(encoding="utf-8"))
        t = rep.get("timings_this_run_s", {})
        runs.append({"papers": rep.get("papers", 0), "total_s": t.get("total_s") or 0.0,
                     "discovery_s": t.get("discovery_s") or 0.0,
                     "pacing": rep.get("arxiv_pacing")})

    R = {"name": d.name, "n_papers": len(papers), "runs": runs}
    R["n_versions"] = sum(int(p["n_versions"]) for p in papers.values())
    R["versions_per_paper"] = [int(p["n_versions"]) for p in papers.values()]
    R["v_status"] = {}
    complete = with_tex = 0
    raw_pp, kept_pp_all, kept_pp_meas, secs_pp = [], [], [], []
    raw_sum = kept_sum = dl_sum = 0.0
    n_ok = n_ok_meas = n_ok_rec = 0
    R["size_time"] = [(_f(r["download_bytes"]), _f(r["seconds"])) for r in dl.values()
                      if r["status"] == "ok" and _f(r["download_bytes"]) and _f(r["seconds"])]
    for aid, p in papers.items():
        rows = [dl.get((aid, str(v))) for v in range(1, int(p["n_versions"]) + 1)]
        for r in rows:
            if r:
                R["v_status"][r["status"]] = R["v_status"].get(r["status"], 0) + 1
        if rows and all(r and r["status"] in FINAL for r in rows):
            complete += 1
        ok = [r for r in rows if r and r["status"] == "ok"]
        if ok:
            with_tex += 1
            kept_pp_all.append(sum(_f(r["kept_bytes"]) or 0 for r in ok))
            n_ok += len(ok)
            meas = [r for r in ok if _f(r["raw_bytes"]) and RECOVERED not in (r.get("note") or "")]
            n_ok_meas += len(meas)
            n_ok_rec += len(ok) - len(meas)
            raw_sum += sum(_f(r["raw_bytes"]) for r in meas)
            kept_sum += sum(_f(r["kept_bytes"]) or 0 for r in meas)
            dl_sum += sum(_f(r["download_bytes"]) or 0 for r in meas)
            if len(meas) == len(ok):                      # every ok version measured
                raw_pp.append(sum(_f(r["raw_bytes"]) for r in ok))
                kept_pp_meas.append(sum(_f(r["kept_bytes"]) or 0 for r in ok))
        if rows and all(r and _f(r["seconds"]) for r in rows):
            secs_pp.append(sum(_f(r["seconds"]) for r in rows))
    R.update(papers_complete=complete, papers_with_tex=with_tex,
             raw_pp=raw_pp, kept_pp_meas=kept_pp_meas, kept_pp_all=kept_pp_all, secs_pp=secs_pp,
             raw_sum=raw_sum, kept_sum=kept_sum, dl_sum=dl_sum,
             n_ok=n_ok, n_ok_measured=n_ok_meas, n_ok_recovered=n_ok_rec)

    R["s2_status"], refs_all, refs_arxiv = {}, [], []
    for aid, r in s2.items():
        if aid in papers:
            R["s2_status"][r["status"]] = R["s2_status"].get(r["status"], 0) + 1
            if r["status"] == "ok":
                refs_all.append(int(_f(r["n_refs"]) or 0))
                refs_arxiv.append(int(_f(r["n_refs_arxiv"]) or 0))
    R.update(refs_all=refs_all, refs_arxiv=refs_arxiv)

    rss = [(_f(r["elapsed_s"]), _f(r["rss_mb"])) for r in res if _f(r["rss_mb"]) is not None]
    out = [(_f(r["elapsed_s"]), _f(r["output_mb"])) for r in res if _f(r.get("output_mb")) is not None]
    R.update(rss=rss, out=out,
             peak_ram=max((v for _, v in rss), default=0.0),
             mean_ram=float(np.mean([v for _, v in rss])) if rss else 0.0,
             peak_disk=max((v for _, v in out), default=0.0))
    R["wall_s"] = sum(r["total_s"] for r in runs)
    R["discovery_s"] = sum(r["discovery_s"] for r in runs)
    R["pacing"] = sorted({r["pacing"] for r in runs if r["papers"]})
    return R


def load_all(range_logs_dir=None):
    here = Path(__file__).resolve().parent
    candidates = [range_logs_dir] if range_logs_dir else []
    candidates += [
        "range_logs",
        "../range_logs",
        str(here / "range_logs"),
        str(here.parent / "range_logs"),
        "../logs_or_benchmarks/range_logs",
        "./logs_or_benchmarks/range_logs",
    ]
    for c in candidates:
        if c and Path(c).exists():
            found = {p.name: load_range(p) for p in sorted(Path(c).glob("r[0-9]*")) if (p / "manifest").exists()}
            if found:
                return found
    raise FileNotFoundError(f"Could not find range_logs/r1..r5 in any of: {candidates}")


def _pooled(S, key):
    return [x for R in S.values() for x in R[key]]


def tables(S):
    """Return the DataFrames used in the report."""
    P = lambda key: sum(R[key] for R in S.values())
    n_p, n_v = P("n_papers"), P("n_versions")
    complete, tex = P("papers_complete"), P("papers_with_tex")
    vs = {}
    for R in S.values():
        for k, v in R["v_status"].items():
            vs[k] = vs.get(k, 0) + v
    s2 = {}
    for R in S.values():
        for k, v in R["s2_status"].items():
            s2[k] = s2.get(k, 0) + v
    raw_pp, kept_pp = _pooled(S, "raw_pp"), _pooled(S, "kept_pp_meas")
    secs = np.array(_pooled(S, "secs_pp"))
    refs, refs_a = _pooled(S, "refs_all"), _pooled(S, "refs_arxiv")
    rawS, keptS, dlS = P("raw_sum"), P("kept_sum"), P("dl_sum")
    walls = [R["wall_s"] for R in S.values()]

    overview = pd.DataFrame([
        ("Papers assigned / versions discovered", f"{n_p:,} / {n_v:,} (avg {n_v / n_p:.2f} versions per paper)"),
        ("Paper completion rate (all versions reached a final status)", f"{complete:,} ({100 * complete / n_p:.2f}%)"),
        ("Paper TeX availability (at least one version with .tex)", f"{tex:,} ({100 * tex / n_p:.2f}%)"),
        ("Version-level TeX extraction rate", f"{vs.get('ok', 0):,} ({100 * vs.get('ok', 0) / n_v:.2f}%)"),
        ("Version-level failures (retryable errors)", f"{vs.get('error', 0):,} ({100 * vs.get('error', 0) / n_v:.2f}%)"),
        ("Avg paper size before / after removing figures",
         f"{np.mean(raw_pp) / MB:.2f} MB / {np.mean(kept_pp) / 1024:.0f} KB  (measured on {len(raw_pp):,} papers)"),
        ("Storage reduction", f"{100 * (1 - keptS / rawS):.2f}%  ({rawS / GB:.1f} GB -> {keptS / GB:.2f} GB)"),
        ("Semantic Scholar: papers indexed / not indexed", f"{s2.get('ok', 0):,} / {s2.get('not_found', 0):,}"),
        ("Semantic Scholar: reference crawl completion", f"{100 * (s2.get('ok', 0) + s2.get('not_found', 0)) / n_p:.2f}%"),
        ("Avg references per paper (all / with arXiv ID)", f"{np.mean(refs):.2f} / {np.mean(refs_a):.2f}"),
        ("Processing time per paper (mean / median / p95)",
         f"{secs.mean():.2f} s / {np.median(secs):.2f} s / {np.percentile(secs, 95):.2f} s  (n = {len(secs):,})"),
        ("End-to-end wall time (parallel: slowest range / serial: sum)",
         f"{max(walls) / 3600:.2f} h / {sum(walls) / 3600:.2f} h"),
        ("Peak / mean RAM (per Colab instance, worst range)",
         f"{max(R['peak_ram'] for R in S.values()):.1f} MB / {np.mean([R['mean_ram'] for R in S.values()]):.1f} MB"),
        ("Peak disk per instance (worst range) / combined final output",
         f"{max(R['peak_disk'] for R in S.values()) / 1024:.2f} GB / {sum(R['peak_disk'] for R in S.values()) / 1024:.2f} GB"),
    ], columns=["Metric", "Value"])

    rows = []
    for R in S.values():
        v = R["v_status"]
        rows.append({
            "Range": R["name"], "Papers": R["n_papers"], "Versions": R["n_versions"],
            "Complete %": round(100 * R["papers_complete"] / R["n_papers"], 2),
            "With TeX %": round(100 * R["papers_with_tex"] / R["n_papers"], 2),
            "ok": v.get("ok", 0), "pdf_only": v.get("pdf_only", 0), "no_source": v.get("no_source", 0),
            "non_tex": v.get("non_tex", 0), "error": v.get("error", 0),
            "S2 not found": R["s2_status"].get("not_found", 0),
            "Avg refs": round(float(np.mean(R["refs_all"])), 1),
            "Reduction %": round(100 * (1 - R["kept_sum"] / R["raw_sum"]), 1) if R["raw_sum"] else None,
            "Measured ok versions %": round(100 * R["n_ok_measured"] / max(1, R["n_ok"]), 1),
            "Mean s/paper": round(float(np.mean(R["secs_pp"])), 2),
            "Median s/paper": round(float(np.median(R["secs_pp"])), 2),
            "Wall h": round(R["wall_s"] / 3600, 2), "Discovery s": round(R["discovery_s"], 1),
            "Peak RAM MB": round(R["peak_ram"], 1), "Mean RAM MB": round(R["mean_ram"], 1),
            "Peak disk GB": round(R["peak_disk"] / 1024, 2), "Pacing": "/".join(R["pacing"]),
        })
    per_range = pd.DataFrame(rows).set_index("Range")
    status = pd.DataFrame({"versions": vs, "share %": {k: round(100 * x / n_v, 2) for k, x in vs.items()}})
    st = np.array(_pooled(S, "size_time"), dtype=float)
    edges = [0, 1e6, 5e6, 20e6, np.inf]
    labels = ["< 1 MB", "1-5 MB", "5-20 MB", "> 20 MB"]
    bucket = pd.cut(st[:, 0], edges, labels=labels, right=False)
    by_size = (pd.DataFrame({"Archive size": bucket, "s": st[:, 1]}).groupby("Archive size", observed=True)["s"]
               .agg(versions="count", mean_s="mean", median_s="median").round(2))
    return {"overview": overview, "per_range": per_range, "status": status, "by_size": by_size}


# ---------------------------------------------------------------- figures
def _style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def plot_versions(S):
    _style()
    vc = pd.Series(_pooled(S, "versions_per_paper")).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(vc.index.astype(str), vc.values, color=COLORS[0], edgecolor="black", alpha=0.85)
    ax.set_title("Versions per paper")
    ax.set_xlabel("Number of versions"), ax.set_ylabel("Papers")
    ax.set_yscale("log"), ax.grid(axis="y", alpha=0.3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=8)
    return fig


def plot_sizes(S):
    _style()
    raw = [b / 1024 for b in _pooled(S, "raw_pp") if b > 0]
    kept = [b / 1024 for b in _pooled(S, "kept_pp_meas") if b > 0]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.logspace(np.log10(max(0.1, min(kept))), np.log10(max(raw)), 40)
    ax.hist(raw, bins=bins, alpha=0.6, color="#ef4444", label=f"Before (mean {np.mean(raw) / 1024:.2f} MB)")
    ax.hist(kept, bins=bins, alpha=0.7, color="#10b981", label=f"After (mean {np.mean(kept):.0f} KB)")
    ax.set_xscale("log"), ax.set_xlabel("Size per paper, KB (log scale)"), ax.set_ylabel("Papers")
    ax.set_title("Paper size before vs. after removing figures"), ax.legend(), ax.grid(alpha=0.3)
    return fig


def plot_refs(S):
    _style()
    a, b = _pooled(S, "refs_all"), _pooled(S, "refs_arxiv")
    top = min(150, int(np.percentile(a, 99)) + 10)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(0, top, 35)
    ax.hist(np.clip(a, 0, top), bins=bins, alpha=0.6, color="#6366f1", label=f"All references (mean {np.mean(a):.1f})")
    ax.hist(np.clip(b, 0, top), bins=bins, alpha=0.75, color="#f59e0b", label=f"With arXiv ID (mean {np.mean(b):.1f})")
    ax.set_xlabel("References per paper (clipped at 99th percentile)"), ax.set_ylabel("Papers")
    ax.set_title("Crawled references per paper"), ax.legend(), ax.grid(alpha=0.3)
    return fig


def plot_latency(S):
    _style()
    s = np.array(_pooled(S, "secs_pp"))
    top = max(15.0, float(np.percentile(s, 98)))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(np.clip(s, 0, top), bins=35, color="#8b5cf6", edgecolor="black", alpha=0.8)
    ax.axvline(s.mean(), color="#ef4444", ls="--", label=f"Mean {s.mean():.2f} s")
    ax.axvline(np.median(s), color="#10b981", label=f"Median {np.median(s):.2f} s")
    ax.set_xlabel("Seconds per paper (all versions, clipped at 98th percentile)"), ax.set_ylabel("Papers")
    ax.set_title("Per-paper processing time"), ax.legend(), ax.grid(alpha=0.3)
    return fig


def _series_plot(S, key, ylabel, title, scale=1.0):
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for c, R in zip(COLORS, S.values()):
        pts = R[key]
        ax.plot([t / 3600 for t, _ in pts], [v / scale for _, v in pts], lw=1, color=c, label=R["name"])
    ax.set_xlabel("Elapsed time within a run (hours)"), ax.set_ylabel(ylabel)
    ax.set_title(title), ax.legend(ncol=5, fontsize=8), ax.grid(alpha=0.3)
    return fig


def plot_ram(S):
    return _series_plot(S, "rss", "RSS (MB)", "Scraper memory over time (one line per Colab instance)")


def plot_disk(S):
    return _series_plot(S, "out", "Output folder (GB)", "Output size over time (one line per instance)", scale=1024)
