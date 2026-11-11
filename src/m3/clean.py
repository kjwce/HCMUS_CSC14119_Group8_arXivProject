"""Deterministic text and identifier cleaning for reference matching."""

from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import unquote


_LATEX_COMMAND = re.compile(r"\\(?:emph|textit|textbf|textrm|texttt|mathrm|mathbf)\s*\{([^{}]*)\}")
_OTHER_COMMAND = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?")
_TOKEN = re.compile(r"[a-z0-9]+")
_MODERN_ARXIV = re.compile(r"(?<!\d)(\d{4})[.\-](\d{4,5})(?:v\d+)?(?!\d)", re.I)
_OLD_ARXIV = re.compile(
    r"(?:arxiv\s*:\s*)?([a-z][a-z0-9.\-]+(?:\/[a-z][a-z0-9.\-]+)?\/\d{7})(?:v\d+)?",
    re.I,
)
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", re.I)

_TITLE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)


def normalize_arxiv_id(value: object) -> str | None:
    """Return an unversioned canonical arXiv identifier.

    Modern identifiers use the public dotted form (``2603.05580``).  Old-style
    identifiers keep their archive prefix and slash.
    """

    if value is None:
        return None
    text = unquote(str(value)).strip()
    text = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text, flags=re.I)
    text = re.sub(r"\.pdf$", "", text, flags=re.I)
    text = re.sub(r"^arxiv\s*:\s*", "", text, flags=re.I)
    modern = _MODERN_ARXIV.fullmatch(text)
    if modern:
        return f"{modern.group(1)}.{modern.group(2)}"
    old = _OLD_ARXIV.fullmatch(text)
    if old:
        return old.group(1).lower()
    return None


def extract_arxiv_id(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        text = unquote(str(value))
        direct = normalize_arxiv_id(text)
        if direct:
            return direct
        modern = _MODERN_ARXIV.search(text)
        if modern:
            return f"{modern.group(1)}.{modern.group(2)}"
        old = _OLD_ARXIV.search(text)
        if old:
            return old.group(1).lower()
    return None


def normalize_doi(value: object) -> str | None:
    if value is None:
        return None
    text = unquote(str(value)).strip().lower()
    text = _DOI_PREFIX.sub("", text).strip().rstrip(".,;)")
    return text or None


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(value: object) -> str:
    """Lowercase and remove non-semantic LaTeX/Unicode variation."""

    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("~", " ").replace("\\&", " and ")
    previous = None
    while previous != text:
        previous = text
        text = _LATEX_COMMAND.sub(r"\1", text)
    text = _OTHER_COMMAND.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = _strip_accents(text).lower()
    return " ".join(_TOKEN.findall(text))


def title_tokens(value: object) -> frozenset[str]:
    tokens = _TOKEN.findall(normalize_text(value))
    useful = [token for token in tokens if token not in _TITLE_STOPWORDS]
    return frozenset(useful or tokens)


def normalize_author(value: object) -> str:
    """Normalize one author while retaining token order for diagnostics."""

    text = normalize_text(value)
    return " ".join(token for token in text.split() if len(token) > 1)


def author_surnames(values: tuple[str, ...] | list[str]) -> frozenset[str]:
    """Extract comparable surname-like tokens from an author list."""

    result: set[str] = set()
    particles = {"da", "de", "del", "der", "di", "la", "le", "van", "von"}
    for value in values:
        normalized = normalize_author(value)
        tokens = normalized.split()
        if not tokens:
            continue
        surname = tokens[-1]
        if len(tokens) >= 2 and tokens[-2] in particles:
            surname = f"{tokens[-2]} {surname}"
        result.add(surname)
    return frozenset(result)
