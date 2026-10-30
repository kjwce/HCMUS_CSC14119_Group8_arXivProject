"""
Milestone 1 — arXiv scraper (single file)

Crawls one assigned arXiv ID range:
  * LaTeX source of EVERY version: only the .tex and .bib files are extracted,
  * metadata.json per paper (title, authors, submission/revised dates, venue),
  * references.json per paper from Semantic Scholar (ALL references, arXiv ID when known).

Run (edit the CONFIG block first, or pass --start/--end):
    pip install requests psutil tqdm
    python scraper_m1.py
    python scraper_m1.py --limit 20          # quick test on the first 20 papers

Stop with Ctrl+C at any time; running the same command again resumes.

Output (default ./data):
    data/raw/2605-27338/{v1,v2,...}/   the .tex and .bib files of each version
    data/raw/2605-27338/metadata.json
    data/raw/2605-27338/references.json
    data/manifest/papers.jsonl         discovery result (versions, dates, metadata)
    data/logs/download_log.csv         one row per version: status, sizes, time
    data/logs/s2_log.csv               one row per paper: Semantic Scholar status
    data/logs/resource_log.csv         RAM / output size over time
    data/logs/report_latest.json       whole-range statistics for the report

Design notes
  * One HTTP client for every arXiv endpoint (API, OAI-PMH, /src/), so the
    3 s politeness gap is shared; one connection at a time.
  * Discovery: arXiv API in batches of 100 IDs (versions, v1 and last-version
    dates). Middle-version dates come from OAI-PMH (arXivRaw). If the API is
    overloaded (429/503), OAI-PMH takes over for the remaining papers.
  * Download: payload type is sniffed from its bytes (gzip/tar, single .tex,
    PDF, .docx/.doc/PostScript, HTML error page); archives are unpacked in
    memory and only .tex/.bib files are written, so figures never touch the disk. Non-TeX versions and HTTP 404
    (e.g. withdrawal notices) are kept as EMPTY version folders.
  * Semantic Scholar runs in a background thread (different host), uses the
    batch endpoint (100 papers per call), adapts its delay on 429 and retries
    failed batches in two slower rounds at the end.
"""
import argparse
import csv
import gzip
import io
import json
import os
import re
import shutil
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

import psutil
import requests
from tqdm import tqdm

# ===========================================================================
# CONFIG — edit here
# ===========================================================================
START_ID = "2605.27338"            # first ID of the assigned range (inclusive)
END_ID = "2606.00733"              # last ID of the assigned range (inclusive)
MONTH_END = {"2605": 31604}        # known last ID of a month; unknown months are searched automatically
FAST_PACING = False                # True: count the 3 s gap from the START of the previous arXiv call
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

USER_AGENT = os.environ.get("CRAWLER_USER_AGENT",
                            "HCMUS-IntroDS-arXiv-crawler/1.0 (student course project)")
# Semantic Scholar key: env var S2_API_KEY, or a file s2_api_key.txt next to
# this script. Leave both empty to run without a key (slower, many 429).
S2_KEY_FILE = Path(__file__).resolve().parent / "s2_api_key.txt"

ARXIV_DELAY = 3.0            # arXiv etiquette: >= 3 s between calls, one connection
ARXIV_MAX_RETRIES = 6
API_RETRIES = 3              # quick retries per API call
API_COOLDOWN = 60            # pause before one more try when the arXiv API is busy
API_SKIP_SECONDS = 600       # then use OAI-PMH for this long before probing the API again
API_BATCH = 100              # IDs per arXiv API call
S2_DELAY = 1.05              # Semantic Scholar: 1 request / second
S2_MAX_DELAY = 5.0
S2_MAX_RETRIES = 12
S2_BATCH_SIZE = 100
S2_RETRY_DELAYS = (2.5, 4.0)  # slower end-of-run retry rounds
S2_RETRY_BATCH = 20
TIMEOUT = 120
DATE_ONLY = True             # dates as "YYYY-MM-DD" (False: full ISO timestamp)

# Figure removal "by extraction": only .tex and .bib files are extracted from
# each source archive (statement, Requirement), so figure binaries and other
# non-source files never enter the corpus. LaTeX figure commands stay inside
# the .tex files untouched.
KEEP_EXTS = {".tex", ".bib"}
BINARY_SNIFF_BYTES = 8192

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_SRC = "https://export.arxiv.org/src/{id}"
ARXIV_OAI = "https://oaipmh.arxiv.org/oai"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper/"

# Semantic Scholar reports the preprint server itself as "venue" for many papers.
NOT_A_VENUE = {"arxiv.org", "arxiv", "arxiv (cornell university)"}


# ===========================================================================
# SMALL HELPERS
# ===========================================================================
def s2_api_key():
    key = os.environ.get("S2_API_KEY", "").strip()
    if not key and S2_KEY_FILE.exists():
        key = S2_KEY_FILE.read_text(encoding="utf-8").strip()
    return key


def folder_name(arxiv_id):
    """'2605.27338' -> '2605-27338' (convention from the statement)."""
    return arxiv_id.replace(".", "-").replace("/", "_")


def ref_key(arxiv_id):
    """references.json key: 'yymm-id' for new IDs, original form for old ones."""
    return arxiv_id.replace(".", "-") if "/" not in arxiv_id else arxiv_id


def iso_from_atom(value):
    if not value:
        return None
    return value[:10] if DATE_ONLY else value


def iso_from_rfc2822(value):
    if not value:
        return None
    dt = parsedate_to_datetime(value).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d") if DATE_ONLY else dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class CsvLog:
    """Append-only CSV log flushed after every row (safe to resume)."""

    def __init__(self, path, fields):
        self.path, self.fields = Path(path), fields
        self._lock = threading.Lock()
        new = not self.path.exists()
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row):
        with self._lock:
            self._writer.writerow({k: row.get(k, "") for k in self.fields})
            self._fh.flush()

    def close(self):
        self._fh.close()


def read_latest(path, key_fields):
    """Last row per key of a CsvLog (later rows win)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {tuple(r[k] for k in key_fields): r for r in csv.DictReader(fh)}


# ===========================================================================
# HTTP CLIENT (shared gap, retry/backoff, adaptive delay)
# ===========================================================================
class PoliteClient:
    """Serialises requests to one host and keeps `delay` seconds between them.

    pacing="end"  : gap counted from the end of the previous response (strict).
    pacing="start": gap counted from when the previous request was sent, so a
                    slow download already counts toward the 3 s (faster).
    adaptive      : 429 -> delay x1.25 (up to max_delay); 20 successes -> x0.9.
    """

    def __init__(self, name, delay, max_retries, headers=None, pacing="end",
                 adaptive=False, max_delay=5.0):
        self.name = name
        self.base_delay = self.delay = delay
        self.max_retries = max_retries
        self.pacing, self.adaptive, self.max_delay = pacing, adaptive, max_delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.session.headers.update(headers or {})
        self._last = 0.0
        self._streak = 0
        self._lock = threading.Lock()

    def _adapt(self, ok):
        if not self.adaptive:
            return
        if ok:
            self._streak += 1
            if self._streak >= 20:
                self.delay, self._streak = max(self.base_delay, self.delay * 0.9), 0
        else:
            self._streak, self.delay = 0, min(self.max_delay, self.delay * 1.25)

    def request(self, method, url, accept=(200,), stop_event=None, retries=None, **kwargs):
        last_error = None
        retries = retries or self.max_retries
        for attempt in range(retries):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("stopped")
            with self._lock:
                wait = self._last + self.delay - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                if self.pacing == "start":
                    self._last = time.monotonic()
                try:
                    resp = self.session.request(method, url, timeout=TIMEOUT, **kwargs)
                except requests.RequestException as exc:
                    resp, last_error = None, repr(exc)
                finally:
                    if self.pacing != "start":
                        self._last = time.monotonic()
            if resp is not None and resp.status_code in accept:
                self._adapt(True)
                return resp
            if resp is not None:
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code == 429:
                    self._adapt(False)
            if attempt + 1 == retries:
                break
            backoff = min(300, 5 * 2 ** attempt)
            ra = resp.headers.get("Retry-After") if resp is not None else None
            if ra and ra.isdigit():
                backoff = max(backoff, int(ra))
            tqdm.write(f"  [{self.name}] {last_error}; retry {attempt + 1}/{retries} in {backoff}s "
                       f"(delay {self.delay:.2f}s)")
            time.sleep(backoff)
        raise RuntimeError(f"{self.name}: giving up on {url} ({last_error})")


def make_s2_client(delay=None):
    key = s2_api_key()
    return PoliteClient("S2", delay or S2_DELAY, S2_MAX_RETRIES, {"x-api-key": key} if key else {},
                        pacing="start", adaptive=True, max_delay=S2_MAX_DELAY)


def check_s2_key():
    """Pre-flight key check; the key itself is never printed."""
    key = s2_api_key()
    if not key:
        print("Semantic Scholar: no API key -> shared keyless pool, expect 429 and slower progress.")
        return
    try:
        r = requests.get(f"{S2_PAPER}arXiv:1706.03762", params={"fields": "title"},
                         headers={"x-api-key": key, "User-Agent": USER_AGENT}, timeout=20)
        if r.status_code == 200:
            print("Semantic Scholar key check: OK (key accepted).")
        elif r.status_code in (401, 403):
            print(f"Semantic Scholar key check: KEY REJECTED (HTTP {r.status_code}). Check the key.")
        else:
            print(f"Semantic Scholar key check: HTTP {r.status_code}; continuing.")
    except requests.RequestException as exc:
        print(f"Semantic Scholar key check: network error ({exc!r}); continuing.")


# ===========================================================================
# STEP 1 — ID LIST (handles ranges that cross a month boundary)
# ===========================================================================
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
OAI_NS = {"o": "http://www.openarchives.org/OAI/2.0/", "r": "http://arxiv.org/OAI/arXivRaw/"}


def parse_id(text):
    month, num = text.strip().split(".")
    if len(month) != 4 or not num.isdigit():
        raise ValueError(f"not a new-style arXiv ID: {text!r}")
    return month, int(num)


def next_month(month):
    yy, mm = int(month[:2]), int(month[2:])
    return f"{yy + (mm == 12):02d}{mm % 12 + 1:02d}"


def month_end(client, month, known_existing):
    """Largest existing number of `month`: API probing (100 IDs per call), OAI-PMH fallback."""
    lo, hi = known_existing, 99999
    try:
        while hi > lo:
            step = max(1, (hi - lo) // 99)
            probes = sorted(set(range(lo, hi + 1, step)) | {hi})
            resp = client.request("GET", ARXIV_API, retries=API_RETRIES, params={
                "id_list": ",".join(f"{month}.{n:05d}" for n in probes), "max_results": len(probes)})
            found = {int(e.findtext("a:id", "", ATOM).split("/abs/")[1].rsplit("v", 1)[0].split(".")[1])
                     for e in ET.fromstring(resp.content).findall("a:entry", ATOM)
                     if "/abs/" in e.findtext("a:id", "", ATOM)}
            best = max((n for n in probes if n in found), default=lo)
            nxt = [n for n in probes if n > best]
            lo, hi = best, (nxt[0] - 1 if nxt else best)
        return lo
    except RuntimeError:
        print("  arXiv API unavailable, falling back to OAI-PMH binary search", flush=True)
    lo, hi = known_existing, 99999
    while lo < hi:
        mid = (lo + hi + 1) // 2
        resp = client.request("GET", ARXIV_OAI, params={
            "verb": "GetRecord", "identifier": f"oai:arXiv.org:{month}.{mid:05d}", "metadataPrefix": "arXiv"})
        if ET.fromstring(resp.content).find("o:GetRecord", OAI_NS) is not None:
            lo = mid
        else:
            hi = mid - 1
    return lo


def build_ids(client, start, end, cache_file):
    """All IDs from start to end inclusive, zero-padded to 5 digits."""
    cache_file = Path(cache_file)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    cache.update(MONTH_END)
    (m, n), (end_m, end_n) = parse_id(start), parse_id(end)
    ids = []
    while m != end_m:
        if m not in cache:
            print(f"Finding the last ID of month {m} ...", flush=True)
            cache[m] = month_end(client, m, n)
            cache_file.write_text(json.dumps(cache, indent=2))
            print(f"  month {m} ends at {m}.{cache[m]:05d}", flush=True)
        ids += [f"{m}.{i:05d}" for i in range(n, cache[m] + 1)]
        m, n = next_month(m), 1
        if m > end_m:
            raise ValueError("END_ID is before START_ID")
    ids += [f"{m}.{i:05d}" for i in range(n, end_n + 1)]
    return ids


# ===========================================================================
# STEP 2 — ENTRY DISCOVERY (versions, dates, metadata)
# ===========================================================================
def parse_api_entry(entry):
    full_id = entry.findtext("a:id", "", ATOM)
    if "/abs/" not in full_id:
        return None
    base, version = full_id.split("/abs/")[1].rsplit("v", 1)
    primary = entry.find("x:primary_category", ATOM)
    return {
        "arxiv_id": base,
        "n_versions": int(version),
        "title": " ".join(entry.findtext("a:title", "", ATOM).split()),
        "authors": [a.findtext("a:name", "", ATOM).strip() for a in entry.findall("a:author", ATOM)],
        "published": entry.findtext("a:published", None, ATOM),   # date of v1
        "updated": entry.findtext("a:updated", None, ATOM),       # date of the last version
        "journal_ref": entry.findtext("x:journal_ref", None, ATOM),
        "doi": entry.findtext("x:doi", None, ATOM),
        "comment": entry.findtext("x:comment", None, ATOM),
        "primary_category": primary.get("term") if primary is not None else None,
        "categories": [c.get("term") for c in entry.findall("a:category", ATOM)],
    }


def fetch_oai_record(client, arxiv_id):
    resp = client.request("GET", ARXIV_OAI, params={
        "verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}", "metadataPrefix": "arXivRaw"})
    return ET.fromstring(resp.content).find(".//r:arXivRaw", OAI_NS)


def oai_versions(raw):
    return [(v.get("version"), iso_from_rfc2822(v.findtext("r:date", None, OAI_NS)))
            for v in raw.findall("r:version", OAI_NS)]


def fetch_oai_paper(client, arxiv_id):
    """Full paper dict from OAI-PMH (used when the arXiv API is unavailable)."""
    raw = fetch_oai_record(client, arxiv_id)
    if raw is None:
        return None
    text = lambda tag: (" ".join((raw.findtext(f"r:{tag}", "", OAI_NS) or "").split()) or None)
    versions = oai_versions(raw)
    categories = (text("categories") or "").split()
    return {
        "arxiv_id": arxiv_id,
        "n_versions": len(versions),
        "title": text("title"),
        "authors": [a.strip() for a in re.split(r",\s*|\s+and\s+", text("authors") or "") if a.strip()],
        "published": None, "updated": None,
        "journal_ref": text("journal-ref"),
        "doi": text("doi"),
        "comment": text("comments"),
        "primary_category": categories[0] if categories else None,
        "categories": categories,
        "version_dates": [d for _, d in versions],
        "source_of_metadata": "oai-pmh",
    }


def version_dates(client, paper):
    """Dates of v1..vN: first/last from the API, middle versions from OAI-PMH."""
    if paper.get("version_dates"):
        return paper["version_dates"]
    n = paper["n_versions"]
    dates = {1: iso_from_atom(paper["published"]), n: iso_from_atom(paper["updated"])}
    if n >= 3:
        raw = fetch_oai_record(client, paper["arxiv_id"])
        if raw is not None:
            dates.update({int(v.lstrip("v")): d for v, d in oai_versions(raw) if int(v.lstrip("v")) <= n})
    return [dates.get(k) for k in range(1, n + 1)]


_api_state = {"skip_until": 0.0}


def fetch_api_batch(client, batch):
    """One arXiv API call for up to 100 IDs; returns {arxiv_id: paper} or None.

    If the API answers 429/503 it gets one cool-down pause and a second try.
    If it is still busy, the API is skipped for API_SKIP_SECONDS (those batches
    use OAI-PMH) and then probed again, so a long outage costs little.
    """
    if time.monotonic() < _api_state["skip_until"]:
        return None
    for k, pause in enumerate((0, API_COOLDOWN)):
        if pause:
            tqdm.write(f"  arXiv API busy; cooling down {pause}s before one more try")
            time.sleep(pause)
        try:
            resp = client.request("GET", ARXIV_API, retries=API_RETRIES,
                                  params={"id_list": ",".join(batch), "max_results": len(batch)})
        except RuntimeError:
            continue
        found = {}
        for entry in ET.fromstring(resp.content).findall("a:entry", ATOM):
            paper = parse_api_entry(entry)
            if paper:
                found[paper["arxiv_id"]] = paper
        return found
    _api_state["skip_until"] = time.monotonic() + API_SKIP_SECONDS
    tqdm.write(f"  arXiv API still busy; using OAI-PMH for the next {API_SKIP_SECONDS // 60} min, then retrying the API")
    return None


def discover(client, ids, manifest_path):
    """Build (or resume) the manifest: one JSON line per paper.

    Works batch by batch (100 IDs) and writes each batch to the manifest
    immediately, so an interruption never loses finished discovery work.
    A batch the arXiv API cannot serve even after cool-downs is read from
    OAI-PMH paper by paper; the next batch tries the API again.
    """
    done = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                done[rec["arxiv_id"]] = rec
    todo = [i for i in ids if i not in done]
    if not todo:
        return [done[i] for i in ids]

    print(f"Discovery: {len(todo)} IDs to look up", flush=True)
    bar = tqdm(total=len(todo), desc="Discovery", unit="paper", dynamic_ncols=True)
    with open(manifest_path, "a", encoding="utf-8") as fh:
        for i in range(0, len(todo), API_BATCH):
            batch = todo[i:i + API_BATCH]
            found = fetch_api_batch(client, batch)
            via_oai = found is None
            if via_oai:
                found = {}
            for arxiv_id in batch:
                paper = found.get(arxiv_id)
                if paper is None and via_oai:
                    paper = fetch_oai_paper(client, arxiv_id)
                if paper is None:
                    rec = {"arxiv_id": arxiv_id, "status": "not_found", "n_versions": 0}
                else:
                    paper["version_dates"] = version_dates(client, paper)
                    rec = {**paper, "status": "ok",
                           "withdrawn": "withdrawn" in (paper["comment"] or "").lower(),
                           "discovered_at": now_iso()}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[arxiv_id] = rec
                bar.update(1)
    bar.close()
    return [done[i] for i in ids]


# ===========================================================================
# STEP 3 — DOWNLOAD + UNPACK + REMOVE FIGURES
# ===========================================================================
TEX_MARKERS = (rb"\documentclass", rb"\documentstyle", rb"\begin{document}", rb"\section",
               rb"\input", rb"\magnification", rb"\bye")
BIB_MARKERS = (b"@article", b"@inproceedings", b"@misc", b"@book", b"@incollection", b"@techreport")
EMPTY_KINDS = {"pdf", "docx", "doc", "ps", "no_tex"}

DL_FIELDS = ["arxiv_id", "version", "status", "http_status", "download_bytes", "raw_bytes",
             "kept_bytes", "n_files", "n_kept", "n_removed", "seconds", "finished_at", "note"]
# Final statuses are skipped on a rerun; "error" rows are retried.
#   ok        source saved (note: tar / single / zip)
#   pdf_only  only a PDF exists                       -> empty vN/ folder
#   non_tex   .docx/.doc/PostScript/no TeX inside      -> empty vN/ folder
#   no_source HTTP 404 (e.g. withdrawal-notice version) -> empty vN/ folder
DL_FINAL = {"ok", "pdf_only", "non_tex", "no_source"}


def safe_relpath(name):
    """Drop absolute prefixes and '..' parts (path traversal guard)."""
    parts = [p for p in PurePosixPath(name.replace("\\", "/")).parts if p not in ("", ".", "..", "/")]
    return Path(*parts) if parts else None


def looks_like_tex(content):
    return any(m in content[:20000] for m in TEX_MARKERS)


def kept_name(relpath, content):
    """Path to save under, or None when the file is not extracted.

    Only text .tex/.bib files are kept. A TeX file shipped without any
    extension (rare, e.g. 'ms') is kept as 'ms.tex' so the paper is not lost.
    """
    if b"\x00" in content[:BINARY_SNIFF_BYTES]:
        return None
    suffix = relpath.suffix.lower()
    if suffix in KEEP_EXTS:
        return relpath
    if suffix == "" and looks_like_tex(content):
        return relpath.with_name(relpath.name + ".tex")
    return None


def non_tex_kind(data):
    """.docx (zip with word/), .doc (OLE2), PostScript: not convertible to TeX."""
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return None
        if "[Content_Types].xml" in names and any(n.startswith("word/") for n in names):
            return "docx"
    if data[:4] == b"\xd0\xcf\x11\xe0":
        return "doc"
    if data[:4] == b"%!PS":
        return "ps"
    return None


def unpack(payload):
    """(kind, [(relpath, bytes), ...]); kind: tar/single/zip/pdf/docx/doc/ps/html/empty/unknown."""
    if not payload:
        return "empty", []
    data = payload
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except OSError:
            return "unknown", []
    if data[:4] == b"%PDF":
        return "pdf", []
    if data.lstrip()[:1] == b"<" and b"<html" in data[:2000].lower():
        return "html", []
    other = non_tex_kind(data)
    if other:
        return other, []
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return "zip", [(rel, zf.read(info)) for info in zf.infolist()
                           if not info.is_dir() and (rel := safe_relpath(info.filename))]
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            return "tar", [(rel, tf.extractfile(m).read()) for m in tf.getmembers()
                           if m.isfile() and (rel := safe_relpath(m.name))]
    except tarfile.TarError:
        pass
    if payload[:2] == b"\x1f\x8b":           # gzip but not tar: single source file
        if b"\x00" in data[:BINARY_SNIFF_BYTES]:
            return "unknown", []
        if looks_like_tex(data):
            name = "main.tex"
        elif any(m in data[:20000].lower() for m in BIB_MARKERS):
            name = "main.bib"
        else:
            name = "main.tex"
        return "single", [(Path(name), data)]
    return "unknown", []


def save_version(payload, target_dir):
    """Unpack, extract only .tex/.bib files, write them atomically to target_dir.

    raw_bytes / kept_bytes are both uncompressed sizes (before / after removing figures).
    """
    kind, files = unpack(payload)
    target_dir = Path(target_dir)
    tmp_dir = target_dir.with_name(target_dir.name + ".part")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    raw_bytes = kept_bytes = n_kept = n_removed = 0
    has_tex = False
    for rel, content in files:
        raw_bytes += len(content)
        name = kept_name(rel, content)
        if name is None:
            n_removed += 1
            continue
        out = tmp_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
        kept_bytes += len(content)
        n_kept += 1
        has_tex = has_tex or name.suffix.lower() == ".tex"
    if files and not has_tex:                 # no .tex at all: not convertible, empty record
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        kind, kept_bytes, n_kept = "no_tex", 0, 0
    shutil.rmtree(target_dir, ignore_errors=True)
    os.replace(tmp_dir, target_dir)
    return {"kind": kind, "raw_bytes": raw_bytes, "kept_bytes": kept_bytes,
            "n_files": len(files), "n_kept": n_kept, "n_removed": n_removed}


def status_of(kind):
    if kind == "pdf":
        return "pdf_only"
    if kind in EMPTY_KINDS:
        return "non_tex"
    return "ok" if kind in ("tar", "single", "zip") else "error"


def download_all(client, papers, raw_dir, log_path, stop_event):
    latest = read_latest(log_path, ["arxiv_id", "version"])
    todo = [(p["arxiv_id"], v) for p in papers for v in range(1, p.get("n_versions", 0) + 1)
            if latest.get((p["arxiv_id"], str(v)), {}).get("status") not in DL_FINAL]
    total = sum(p.get("n_versions", 0) for p in papers)
    print(f"Download: {len(todo)} of {total} versions left (pacing measured from the {client.pacing} "
          f"of the previous call)", flush=True)
    if not todo:
        return
    log = CsvLog(log_path, DL_FIELDS)
    counts = {}
    bar = tqdm(total=len(todo), unit="ver", desc="Sources", dynamic_ncols=True)
    try:
        for arxiv_id, v in todo:
            if stop_event.is_set():
                break
            row, t0 = {"arxiv_id": arxiv_id, "version": v}, time.monotonic()
            target = raw_dir / folder_name(arxiv_id) / f"v{v}"
            try:
                resp = client.request("GET", ARXIV_SRC.format(id=f"{arxiv_id}v{v}"),
                                      accept=(200, 404), stop_event=stop_event)
                row.update(http_status=resp.status_code, download_bytes=len(resp.content))
                if resp.status_code == 404:
                    row.update(save_version(b"", target), status="no_source")
                else:
                    stats = save_version(resp.content, target)
                    row.update(stats, status=status_of(stats["kind"]))
                row["note"] = row.pop("kind", "")
            except InterruptedError:
                break
            except Exception as exc:          # retried next run
                row.update(status="error", note=repr(exc)[:300])
            row.update(seconds=round(time.monotonic() - t0, 3), finished_at=now_iso())
            log.write(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            bar.set_postfix_str(f"{arxiv_id}v{v} " + " ".join(f"{k}={n}" for k, n in sorted(counts.items())))
            bar.update(1)
    finally:
        bar.close()
        log.close()


# ===========================================================================
# STEP 4 — SEMANTIC SCHOLAR REFERENCES (background thread)
# ===========================================================================
S2_FIELDS_LOG = ["arxiv_id", "status", "s2_paper_id", "venue", "n_refs", "n_refs_arxiv", "finished_at"]
S2_FINAL = {"ok", "not_found"}
S2_FIELDS = ("paperId,venue,references.paperId,references.externalIds,references.title,"
             "references.authors,references.publicationDate,references.year")


def build_references(refs):
    """S2 reference list -> references.json: EVERY crawled reference is kept.

    Each entry: title, authors, submission date (S2 publicationDate), arXiv ID
    when one exists (plus its 'yymm-id' key) and the Semantic Scholar ID.
    Only entries with an arXiv ID become rows of the final references table.
    """
    out = []
    for ref in refs or []:
        ref = ref or {}
        arxiv = (ref.get("externalIds") or {}).get("ArXiv")
        date = ref.get("publicationDate")
        if not DATE_ONLY and date:
            date += "T00:00:00Z"
        out.append({
            "title": ref.get("title"),
            "authors": [a.get("name") for a in ref.get("authors") or [] if a.get("name")],
            "submission_date": date,
            "year": ref.get("year"),
            "arxiv_id": arxiv,
            "arxiv_key": ref_key(arxiv) if arxiv else None,
            "semantic_scholar_id": ref.get("paperId"),
        })
    return out


def s2_pending(papers, log_path):
    latest = read_latest(log_path, ["arxiv_id"])
    return [p["arxiv_id"] for p in papers
            if p.get("status") == "ok" and latest.get((p["arxiv_id"],), {}).get("status") not in S2_FINAL]


def s2_pass(client, ids, batch_size, raw_dir, log, stop_event, label):
    for i in range(0, len(ids), batch_size):
        if stop_event.is_set():
            return
        batch = ids[i:i + batch_size]
        try:
            results = client.request("POST", S2_BATCH, params={"fields": S2_FIELDS},
                                     json={"ids": [f"arXiv:{x}" for x in batch]}, stop_event=stop_event).json()
        except InterruptedError:
            return
        except Exception as exc:
            tqdm.write(f"  [S2] batch starting {batch[0]} failed ({exc}); queued for retry")
            for arxiv_id in batch:
                log.write({"arxiv_id": arxiv_id, "status": "error"})
            continue
        for arxiv_id, item in zip(batch, results):
            paper_dir = raw_dir / folder_name(arxiv_id)
            paper_dir.mkdir(parents=True, exist_ok=True)
            if item is None:
                write_json_atomic(paper_dir / "references.json", [])
                log.write({"arxiv_id": arxiv_id, "status": "not_found", "finished_at": now_iso()})
                continue
            refs = item.get("references") or []
            out = build_references(refs)
            write_json_atomic(paper_dir / "references.json", out)
            log.write({"arxiv_id": arxiv_id, "status": "ok", "s2_paper_id": item.get("paperId"),
                       "venue": item.get("venue") or "", "n_refs": len(refs),
                       "n_refs_arxiv": sum(1 for r in out if r["arxiv_id"]), "finished_at": now_iso()})
        tqdm.write(f"  [S2 {label}] {min(i + batch_size, len(ids))}/{len(ids)} papers (delay {client.delay:.2f}s)")


def crawl_references(papers, raw_dir, log_path, stop_event, errors):
    try:
        todo = s2_pending(papers, log_path)
        tqdm.write(f"Semantic Scholar: {len(todo)} papers to query, "
                   f"{'with API key' if s2_api_key() else 'WITHOUT API key (slower)'}")
        log = CsvLog(log_path, S2_FIELDS_LOG)
        try:
            s2_pass(make_s2_client(), todo, S2_BATCH_SIZE, raw_dir, log, stop_event, "main")
            for round_no, delay in enumerate(S2_RETRY_DELAYS, 1):
                left = s2_pending(papers, log_path)
                if not left or stop_event.is_set():
                    break
                tqdm.write(f"Semantic Scholar retry round {round_no}: {len(left)} papers, delay {delay}s")
                s2_pass(make_s2_client(delay), left, S2_RETRY_BATCH, raw_dir, log, stop_event, f"retry {round_no}")
        finally:
            log.close()
    except Exception as exc:
        errors.append(exc)


# ===========================================================================
# STEP 5 — metadata.json
# ===========================================================================
def metadata_record(paper, venue=None, version_status=None):
    dates = paper.get("version_dates") or []
    return {
        "title": paper.get("title"),
        "authors": paper.get("authors", []),
        "submission_date": dates[0] if dates else None,
        "revised_dates": dates[1:],
        "venue": paper.get("journal_ref") or venue or None,
        "arxiv_id": paper["arxiv_id"],
        "n_versions": paper.get("n_versions", 0),
        "versions": [{"version": f"v{i}", "date": d, "source_status": (version_status or {}).get(i)}
                     for i, d in enumerate(dates, 1)],
        "doi": paper.get("doi"),
        "journal_ref": paper.get("journal_ref"),
        "comment": paper.get("comment"),
        "primary_category": paper.get("primary_category"),
        "categories": paper.get("categories", []),
        "withdrawn": paper.get("withdrawn", False),
    }


def finalize_metadata(papers, raw_dir, dl_log, s2_log):
    dl = read_latest(dl_log, ["arxiv_id", "version"])
    s2 = read_latest(s2_log, ["arxiv_id"])
    for p in papers:
        if p.get("status") != "ok":
            continue
        status = {v: dl.get((p["arxiv_id"], str(v)), {}).get("status") for v in range(1, p["n_versions"] + 1)}
        venue = s2.get((p["arxiv_id"],), {}).get("venue") or None
        if venue and venue.strip().lower() in NOT_A_VENUE:
            venue = None
        paper_dir = raw_dir / folder_name(p["arxiv_id"])
        paper_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(paper_dir / "metadata.json", metadata_record(p, venue, status))


# ===========================================================================
# MONITOR + REPORT
# ===========================================================================
def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class ResourceMonitor(threading.Thread):
    """Logs RSS of the process (+children) every 10 s and output size every minute."""

    def __init__(self, log_path, data_dir, interval=10.0):
        super().__init__(daemon=True)
        self.log = CsvLog(log_path, ["time", "elapsed_s", "rss_mb", "output_mb", "disk_free_gb"])
        self.data_dir, self.interval = Path(data_dir), interval
        self.peak_rss_mb = self.peak_output_mb = 0.0
        self._halt = threading.Event()
        self._proc = psutil.Process()
        self._t0 = time.monotonic()

    def sample(self, with_disk=True):
        rss = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        rss_mb = rss / 2**20
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        row = {"time": now_iso(), "elapsed_s": round(time.monotonic() - self._t0, 1), "rss_mb": round(rss_mb, 1)}
        if with_disk:
            out_mb = dir_size(self.data_dir) / 2**20
            self.peak_output_mb = max(self.peak_output_mb, out_mb)
            row.update(output_mb=round(out_mb, 1),
                       disk_free_gb=round(psutil.disk_usage(str(self.data_dir)).free / 2**30, 2))
        self.log.write(row)

    def run(self):
        n = 0
        while not self._halt.wait(self.interval):
            n += 1
            self.sample(with_disk=(n % 6 == 0))

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        self.sample()
        self.log.close()


def build_report(papers, logs_dir, timings, monitor, start, end, fast_pacing):
    """Statistics over the whole range, merging every run (resume-safe)."""
    dl = read_latest(logs_dir / "download_log.csv", ["arxiv_id", "version"])
    s2 = read_latest(logs_dir / "s2_log.csv", ["arxiv_id"])
    raw_dir = logs_dir.parent / "raw"
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    n_papers = len(papers)
    n_versions = sum(p.get("n_versions", 0) for p in papers)
    v_status = {}
    for row in dl.values():
        v_status[row["status"]] = v_status.get(row["status"], 0) + 1
    p_complete = p_src = 0
    raw_pp, kept_pp, secs_pp = [], [], []
    for p in papers:
        rows = [dl.get((p["arxiv_id"], str(v))) for v in range(1, p.get("n_versions", 0) + 1)]
        if rows and all(r and r["status"] in DL_FINAL for r in rows):
            p_complete += 1
        ok_rows = [r for r in rows if r and r["status"] == "ok"]
        if ok_rows:
            p_src += 1
            raw_pp.append(sum(int(r["raw_bytes"] or 0) for r in ok_rows))
            # "after" size measured on disk (auditable, like du) over the kept .tex/.bib files
            kept_pp.append(sum(dir_size(raw_dir / folder_name(p["arxiv_id"]) / f"v{r['version']}")
                               for r in ok_rows))
        if rows and all(rows):
            secs_pp.append(sum(float(r["seconds"] or 0) for r in rows))
    s2_ok = [r for r in s2.values() if r["status"] == "ok"]
    s2_status = {}
    for r in s2.values():
        s2_status[r["status"]] = s2_status.get(r["status"], 0) + 1
    rss, out_mb = [], [monitor.peak_output_mb]
    res_log = logs_dir / "resource_log.csv"
    if res_log.exists():
        with open(res_log, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("rss_mb"):
                    rss.append(float(r["rss_mb"]))
                if r.get("output_mb"):
                    out_mb.append(float(r["output_mb"]))
    return {
        "range": [start, end],
        "generated_at": now_iso(),
        "papers": n_papers,
        "versions": n_versions,
        "papers_complete": p_complete,
        "papers_with_tex_source": p_src,
        "paper_success_rate": round(p_complete / n_papers, 4) if n_papers else 0,
        "version_status_counts": v_status,
        "version_success_rate": round(sum(v_status.get(s, 0) for s in DL_FINAL) / n_versions, 4) if n_versions else 0,
        "avg_paper_bytes_before_removing_figures": round(mean(raw_pp)),
        "avg_paper_bytes_after_removing_figures": round(mean(kept_pp)),
        "size_reduction": round(1 - sum(kept_pp) / sum(raw_pp), 4) if sum(raw_pp) else 0,
        "s2_status_counts": s2_status,
        "s2_success_rate": round(len(s2_ok) / n_papers, 4) if n_papers else 0,
        "avg_references_per_paper": round(mean([int(r["n_refs"] or 0) for r in s2_ok]), 2),
        "avg_arxiv_references_per_paper": round(mean([int(r["n_refs_arxiv"] or 0) for r in s2_ok]), 2),
        "avg_download_seconds_per_paper": round(mean(secs_pp), 2),
        "timings_this_run_s": timings,
        "peak_ram_mb_all_runs": round(max(rss + [monitor.peak_rss_mb]), 1),
        "mean_ram_mb_all_runs": round(mean(rss), 1),
        "peak_output_mb": round(max(out_mb), 1),
        "s2_api_key_used": bool(s2_api_key()),
        "arxiv_pacing": "start" if fast_pacing else "end",
    }


def print_report(rep):
    pct = lambda x: f"{100 * x:.1f}%"
    mb = lambda b: f"{b / 2**20:.2f} MB"
    t = rep["timings_this_run_s"]
    print("\n" + "\n".join([
        "=" * 66,
        f"  MILESTONE 1 — CRAWL REPORT  ({rep['range'][0]} -> {rep['range'][1]})",
        "=" * 66,
        f"  Papers / versions             : {rep['papers']} / {rep['versions']}",
        f"  Papers complete (all versions): {rep['papers_complete']}  ({pct(rep['paper_success_rate'])})",
        f"  Papers with TeX source        : {rep['papers_with_tex_source']}",
        f"  Version status                : {rep['version_status_counts']}",
        f"  Version success rate          : {pct(rep['version_success_rate'])}",
        f"  Avg paper size before / after : {mb(rep['avg_paper_bytes_before_removing_figures'])} / "
        f"{mb(rep['avg_paper_bytes_after_removing_figures'])}  (-{pct(rep['size_reduction'])})",
        f"  Semantic Scholar              : {rep['s2_status_counts']}  ({pct(rep['s2_success_rate'])})",
        f"  Avg refs / paper (all, arXiv) : {rep['avg_references_per_paper']} , {rep['avg_arxiv_references_per_paper']}",
        "-" * 66,
        f"  Wall time this run            : {t.get('total_s', 0) / 3600:.2f} h  (discovery {t.get('discovery_s', 0):.0f} s)",
        f"  Avg download time / paper     : {rep['avg_download_seconds_per_paper']} s",
        f"  RAM peak / mean (all runs)    : {rep['peak_ram_mb_all_runs']} MB / {rep['mean_ram_mb_all_runs']} MB",
        f"  Output size peak              : {rep['peak_output_mb']} MB",
        "=" * 66,
    ]))


# ===========================================================================
# MAIN
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Milestone 1 arXiv scraper (single file).")
    ap.add_argument("--start", default=START_ID, help=f"first arXiv ID (default {START_ID})")
    ap.add_argument("--end", default=END_ID, help=f"last arXiv ID, inclusive (default {END_ID})")
    ap.add_argument("--out", default=str(OUTPUT_DIR), help="output folder (default ./data)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N papers (testing)")
    ap.add_argument("--no-s2", action="store_true", help="skip Semantic Scholar")
    ap.add_argument("--fast-pacing", action="store_true", default=FAST_PACING,
                    help="count the 3 s arXiv gap from the START of the previous call (faster)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out).resolve()
    manifest_dir, logs_dir, raw_dir = out / "manifest", out / "logs", out / "raw"
    for d in (manifest_dir, logs_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)
    dl_log, s2_log = logs_dir / "download_log.csv", logs_dir / "s2_log.csv"

    monitor = ResourceMonitor(logs_dir / "resource_log.csv", out)
    monitor.start()
    t_start, timings, stop = time.monotonic(), {}, threading.Event()
    arxiv = PoliteClient("arXiv", ARXIV_DELAY, ARXIV_MAX_RETRIES,
                         pacing="start" if args.fast_pacing else "end")
    if not args.no_s2:
        check_s2_key()
    papers = []
    try:
        t0 = time.monotonic()
        ids = build_ids(arxiv, args.start, args.end, manifest_dir / "month_end.json")
        if args.limit:
            ids = ids[:args.limit]
        print(f"Range {args.start} -> {args.end}: {len(ids)} IDs", flush=True)
        papers = discover(arxiv, ids, manifest_dir / "papers.jsonl")
        timings["discovery_s"] = round(time.monotonic() - t0, 1)
        print(f"Discovery done in {timings['discovery_s']} s: {len(papers)} papers, "
              f"{sum(p.get('n_versions', 0) for p in papers)} versions", flush=True)
        for p in papers:                       # first metadata.json, refreshed at the end
            d = raw_dir / folder_name(p["arxiv_id"])
            if p.get("status") == "ok" and not (d / "metadata.json").exists():
                d.mkdir(parents=True, exist_ok=True)
                write_json_atomic(d / "metadata.json", metadata_record(p))

        errors, s2_thread = [], None
        if not args.no_s2:
            s2_thread = threading.Thread(target=crawl_references,
                                         args=(papers, raw_dir, s2_log, stop, errors), daemon=True)
            s2_thread.start()
        t0 = time.monotonic()
        download_all(arxiv, papers, raw_dir, dl_log, stop)
        timings["download_s"] = round(time.monotonic() - t0, 1)
        if s2_thread:
            if s2_thread.is_alive():
                print("Sources done; waiting for Semantic Scholar to finish ...", flush=True)
            s2_thread.join()
            timings["download_and_references_s"] = round(time.monotonic() - t0, 1)
        if errors:
            print(f"Semantic Scholar thread failed: {errors[0]!r}", flush=True)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C) ... progress is saved, run the same command again to resume.", flush=True)
        stop.set()
    finally:
        if papers:
            finalize_metadata(papers, raw_dir, dl_log, s2_log)
        timings["total_s"] = round(time.monotonic() - t_start, 1)
        monitor.stop()
        rep = build_report(papers, logs_dir, timings, monitor, args.start, args.end, args.fast_pacing)
        write_json_atomic(logs_dir / f"run_summary_{time.strftime('%Y%m%d_%H%M%S')}.json", rep)
        write_json_atomic(logs_dir / "report_latest.json", rep)
        print_report(rep)


if __name__ == "__main__":
    main()
