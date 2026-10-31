"""Extract bibliography entries per version and deduplicate them per publication.

``extract(version_dir, text)`` reads the ``.bib`` files the compiled document
uses (``\\bibliography``/``\\addbibresource``) and converts ``\\bibitem``
entries written in the TeX source (``thebibliography``) to the same structure.
``dedup(entries_by_version)`` merges entries that describe the same work inside
and across versions, unions their fields, and returns one rename map per
version for ``normalize.rename_cites``.

References used while writing this module:

* BibTeX format: O. Patashnik, *BibTeXing* (1988), https://ctan.org/pkg/bibtex
* bibtexparser 1.x documentation: https://bibtexparser.readthedocs.io/en/v1.4.4/
* natbib/revtex ``.bbl`` conventions (``\\newblock``, ``\\bibinfo``):
  https://ctan.org/pkg/natbib and https://ctan.org/pkg/revtex
* DOI syntax: https://www.doi.org/doi_handbook/2_Numbering.html
* arXiv identifier schemes: https://info.arxiv.org/help/arxiv_identifier.html
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gather import read_tex
from .normalize import cite_keys

LOGGER = logging.getLogger(__name__)

TABLE_COLUMNS = ("arxiv_id", "citation_key", "entry_type", "title", "authors", "year", "doi", "journal", "url")
MERGED_FIELDS = ("title", "year", "doi", "journal", "url", "eprint")


@dataclass(slots=True)
class Entry:
    """One bibliography entry of one version, already converted to plain text."""

    citation_key: str
    entry_type: str = "misc"
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    journal: str | None = None
    url: str | None = None
    eprint: str | None = None  # arXiv identifier of the cited work, if known
    source: str = "bib"  # "bib", "bibitem" or "bbl"
    arxiv_id: str | None = None  # citing publication, e.g. "2605-27338"

    def completeness(self) -> int:
        return sum(
            bool(value)
            for value in (self.title, self.authors, self.year, self.doi, self.journal, self.url, self.eprint)
        )


# ---------------------------------------------------------------------------
# LaTeX to plain text for bibliography fields
# ---------------------------------------------------------------------------

_ACCENTS = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "=": "\u0304",
    ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
    "r": "\u030a", "b": "\u0331", "d": "\u0323",
}
_SPECIAL_LETTERS = {
    "ss": "ß", "o": "ø", "O": "Ø", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "aa": "å",
    "AA": "Å", "l": "ł", "L": "Ł", "i": "ı", "j": "ȷ",
}
_SYMBOL_ACCENT_RE = re.compile(r"\\(['`^\"~=.])\s*(?:\{\s*\\?([A-Za-z])\s*\}|\\?([A-Za-z]))")
_LETTER_ACCENT_RE = re.compile(r"\\([uvHckrbd])(?:\s*\{\s*\\?([A-Za-z])\s*\}|\s+([A-Za-z]))")
_SPECIAL_LETTER_RE = re.compile(r"\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L|i|j)(?![A-Za-z])\s*(?:\{\})?")
_ESCAPED_RE = re.compile(r"\\([&%_#$])")
_DROPPED_COMMANDS_RE = re.compile(
    r"\\(?:newblock|relax|BibitemOpen|BibitemShut\s*\{[^{}]*\}|EndOfBibitem|natexlab\s*\{[^{}]*\}"
    r"|mcite[A-Za-z]*(?:\s*\{[^{}]*\})*|bibAnnoteFile\s*\{[^{}]*\}|bibAnnote\s*\{[^{}]*\}\s*\{[^{}]*\}"
    r"|selectlanguage\s*\{[^{}]*\}|penalty\s*-?\d+|bibitemStop|bibitemNoStop|Bibitem[A-Za-z]*"
    r"|providecommand)(?![A-Za-z])"
)
_COMMAND_RE = re.compile(r"\\[A-Za-z@]+\*?")
_WHITESPACE_RE = re.compile(r"\s+")


def _accent(match: re.Match[str]) -> str:
    letter = match.group(2) or match.group(3)
    return unicodedata.normalize("NFC", letter + _ACCENTS[match.group(1)])


def latex_to_text(value: str | None) -> str:
    """Convert a BibTeX/LaTeX field value to plain Unicode text."""

    if not value:
        return ""
    text = value
    if "\\" in text:
        text = _SYMBOL_ACCENT_RE.sub(_accent, text)
        text = _LETTER_ACCENT_RE.sub(_accent, text)
        text = _SPECIAL_LETTER_RE.sub(lambda match: _SPECIAL_LETTERS[match.group(1)], text)
        text = _ESCAPED_RE.sub(r"\1", text)
        text = _DROPPED_COMMANDS_RE.sub(" ", text)
        text = re.sub(r"\\(?:url|href)\s*\{([^{}]*)\}(?:\s*\{[^{}]*\})?", r"\1", text)
        text = _COMMAND_RE.sub(" ", text)
        text = text.replace("\\\\", " ").replace("\\", "")
    text = text.replace("~", " ").replace("---", "—").replace("--", "–")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("{", "").replace("}", "").replace("$", "")
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _clean_field(value: str | None) -> str | None:
    text = latex_to_text(value).strip(" ,;:.")
    return text or None


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;{}\\\"<>]+", re.IGNORECASE)
_ARXIV_NEW_RE = re.compile(
    r"(?:arxiv(?:\.org/(?:abs|pdf)/|\s*(?:preprint)?\s*(?:e-print)?\s*:?\s*(?:arxiv\s*:)?\s*)|abs/)"
    r"(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_ARXIV_OLD_RE = re.compile(r"arxiv(?:\.org/abs/|\s*:\s*)([a-z\-]+(?:\.[A-Z]{2})?/\d{7})", re.IGNORECASE)
_ARXIV_BARE_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.IGNORECASE)
_URL_RE = re.compile(r"\\url\s*\{([^{}]+)\}|\\href\s*\{([^{}]+)\}|(https?://[^\s{}\\]+)")
_YEAR_RE = re.compile(r"(?<![\d./-])((?:19|20)\d{2})(?![\d/])")


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = _DOI_RE.search(value.replace("\\_", "_"))
    if not match:
        return None
    return match.group(0).rstrip(".").lower()


def normalize_arxiv(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if _ARXIV_BARE_RE.match(value):
        return re.sub(r"v\d+$", "", value).lower()
    match = _ARXIV_NEW_RE.search(value) or _ARXIV_OLD_RE.search(value)
    return match.group(1).lower() if match else None


def parse_year(value: Any) -> int | None:
    if value is None:
        return None
    match = _YEAR_RE.search(str(value))
    return int(match.group(1)) if match else None


def normalize_title(title: str | None) -> str:
    """Matching key for titles: case-, accent- and punctuation-insensitive."""

    if not title:
        return ""
    decomposed = unicodedata.normalize("NFKD", title.lower())
    letters = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", letters))


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------


def _split_top_level(text: str, separator_re: re.Pattern[str]) -> list[str]:
    """Split on ``separator_re`` matches that are not inside braces."""

    parts: list[str] = []
    depth = 0
    start = 0
    position = 0
    while position < len(text):
        char = text[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif depth == 0:
            match = separator_re.match(text, position)
            if match and match.end() > position:
                parts.append(text[start:position])
                start = position = match.end()
                continue
        position += 1
    parts.append(text[start:])
    return parts


_BIBTEX_AND_RE = re.compile(r"\s+and\s+", re.IGNORECASE)


def parse_bibtex_authors(value: str | None) -> list[str]:
    """Split a BibTeX ``author`` field into "Given Family" names."""

    if not value:
        return []
    names: list[str] = []
    for raw in _split_top_level(value, _BIBTEX_AND_RE):
        raw = raw.strip()
        if not raw or raw.lower() in {"others", "et al.", "et al"}:
            continue
        parts = [part.strip() for part in _split_top_level(raw, re.compile(r","))]
        if len(parts) == 2:
            raw = f"{parts[1]} {parts[0]}"
        elif len(parts) >= 3:  # "von Last, Jr, First"
            raw = f"{parts[2]} {parts[0]} {parts[1]}"
        name = latex_to_text(raw)
        if name:
            names.append(name)
    return names


_INITIALS_RE = re.compile(r"^(?:[A-Z][a-z]?\.?[\s-]*)+$")
_FREE_AND_RE = re.compile(r"\s*(?:,\s*and\s+|\band\s+|\s+&\s+|\\&\s*|;\s*)", re.IGNORECASE)


def parse_free_authors(text: str) -> list[str]:
    """Split a free-text author list from a ``\\bibitem``.

    Handles "A. B. Smith, C. Jones, and D. Lee", "Smith, A., Jones, B. and Lee, C."
    and "Smith, A.; Jones, B.".
    """

    text = latex_to_text(text)
    text = re.sub(r"\bet\s+al\.?", "", text)
    text = re.sub(r"\.(?:\s*\.)+", ".", text).strip(" ,;")
    if not text:
        return []
    chunks = [chunk.strip(" ,") for chunk in _FREE_AND_RE.split(text)]
    pieces: list[str] = []
    for chunk in chunks:
        pieces.extend(piece.strip() for piece in chunk.split(",") if piece.strip())
    names: list[str] = []
    index = 0
    while index < len(pieces):
        piece = pieces[index]
        following = pieces[index + 1] if index + 1 < len(pieces) else ""
        # "Surname, I. J." is written as two comma-separated pieces.
        if following and _INITIALS_RE.match(following) and not _INITIALS_RE.match(piece):
            names.append(f"{following.strip()} {piece}")
            index += 2
            continue
        names.append(piece)
        index += 1
    # Keep the period of a final initial ("A. D.") but not a sentence period.
    names = [name[:-1] if re.search(r"[A-Za-z]{2,}\.$", name) else name for name in names]
    return [name for name in names if 1 < len(name) <= 80]


# ---------------------------------------------------------------------------
# .bib files
# ---------------------------------------------------------------------------

_BIBLIOGRAPHY_RE = re.compile(r"\\(?:bibliography|addbibresource|addglobalbib|nobibliography)\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
_NOCITE_ALL_RE = re.compile(r"\\nocite\s*\{\s*\*\s*\}")


_PARSER: Any = None


def _bibtexparser_loads(text: str) -> list[dict[str, str]]:
    """Parse with one reused bibtexparser instance per process.

    Building a ``BibTexParser`` compiles its pyparsing grammar, which costs more
    than parsing a typical paper's bibliography. The grammar only reads
    ``parser.bib_database`` when it runs, so a fresh database per call is enough.
    """

    global _PARSER
    from bibtexparser.bibdatabase import BibDatabase

    if _PARSER is None:
        from bibtexparser.bparser import BibTexParser

        # Files that redefine common @string macros (e.g. "cacm") make
        # bibtexparser log one warning per macro; they are harmless.
        logging.getLogger("bibtexparser").setLevel(logging.ERROR)
        _PARSER = BibTexParser(common_strings=True, ignore_nonstandard_types=False, interpolate_strings=True)
        _PARSER.expect_multiple_parse = True
    _PARSER.bib_database = BibDatabase()
    _PARSER.bib_database.load_common_strings()
    return _PARSER.parse(text).entries


# Keys may contain parentheses ("statlog_(german_credit_data)_144").
_ANY_ENTRY_RE = re.compile(r"@\s*([A-Za-z]+)\s*[{(]\s*([^,\s{}]*)")


def select_entries(text: str, keys: set[str] | None) -> str:
    """Keep only ``@string``/``@preamble`` blocks and entries whose key is in ``keys``.

    Shared ``.bib`` files can hold thousands of entries while a paper cites a
    few dozen; parsing only those is 5-20x faster.
    ``keys=None`` keeps the whole file (``\\nocite{*}``).
    """

    if keys is None:
        return text
    starts = list(_ANY_ENTRY_RE.finditer(text))
    kept: list[str] = []
    for index, match in enumerate(starts):
        entry_type = match.group(1).lower()
        if entry_type == "comment":
            continue
        if entry_type in {"string", "preamble"} or match.group(2).strip() in keys:
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            kept.append(text[match.start() : end])
    return "\n".join(kept)


_AT_RE = re.compile(r"@\s*([A-Za-z]+)\s*([{(])")
_BRACE_RE = re.compile(r"[{}]")
_QUOTE_RE = re.compile(r'[{}"]')
_KEY_RE = re.compile(r"\s*([^,\s{}]*)\s*")
_LINE_COMMENT_RE = re.compile(r"%[^\n]*")
_FIELD_NAME_RE = re.compile(r"[\s,]*([^\s=,{}()\"#]+)\s*=\s*")
_BARE_VALUE_RE = re.compile(r"[^\s,#{}()\"]+")
_CONCAT_RE = re.compile(r"\s*#\s*")
_MONTH_MACROS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April", "may": "May",
    "jun": "June", "jul": "July", "aug": "August", "sep": "September", "oct": "October",
    "nov": "November", "dec": "December",
}


class _BibSyntaxError(ValueError):
    pass


def _skip_braced(text: str, start: int) -> int:
    """Return the index just after the ``}`` that closes the ``{`` at ``start``."""

    depth = 0
    for match in _BRACE_RE.finditer(text, start):
        depth += 1 if match.group(0) == "{" else -1
        if depth == 0:
            return match.end()
    raise _BibSyntaxError("unbalanced braces")


def _read_value(text: str, position: int, strings: dict[str, str]) -> tuple[str, int]:
    """Read ``{..}``, ``".."``, a number or a macro, joined with ``#``."""

    parts: list[str] = []
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            raise _BibSyntaxError("value expected")
        char = text[position]
        if char == "{":
            end = _skip_braced(text, position)
            parts.append(text[position + 1 : end - 1])
            position = end
        elif char == '"':
            depth = 0
            for match in _QUOTE_RE.finditer(text, position + 1):
                token = match.group(0)
                if token == "{":
                    depth += 1
                elif token == "}":
                    depth -= 1
                elif depth == 0:
                    parts.append(text[position + 1 : match.start()])
                    position = match.end()
                    break
            else:
                raise _BibSyntaxError("unterminated string")
        else:
            bare = _BARE_VALUE_RE.match(text, position)
            if bare is None:
                raise _BibSyntaxError("value expected")
            word = bare.group(0)
            parts.append(strings.get(word.lower(), word))
            position = bare.end()
        concat = _CONCAT_RE.match(text, position)
        if not concat or concat.end() == position:
            return "".join(parts), position
        position = concat.end()


def _parse_entry(
    text: str, at: re.Match[str], strings: dict[str, str]
) -> tuple[dict[str, str] | None, int]:
    """Parse the entry starting at ``at``; return ``(record or None, end)``."""

    entry_type = at.group(1).lower()
    closer = "}" if at.group(2) == "{" else ")"
    position = at.end()
    if entry_type in {"comment", "preamble"}:
        return None, _skip_braced(text, at.end() - 1) if closer == "}" else position
    if entry_type == "string":
        name = _FIELD_NAME_RE.match(text, position)
        if name is None:
            raise _BibSyntaxError("bad @string")
        value, position = _read_value(text, name.end(), strings)
        strings[name.group(1).lower()] = value
        return None, text.find(closer, position) + 1 or len(text)
    key = _KEY_RE.match(text, position)
    entry = {"ENTRYTYPE": entry_type, "ID": key.group(1)}
    position = key.end()
    while True:
        while position < len(text) and (text[position].isspace() or text[position] in ",%"):
            if text[position] == "%":
                # Authors comment out fields with "%"; skip the line.
                position = _LINE_COMMENT_RE.match(text, position).end()
            else:
                position += 1
        if position >= len(text):
            raise _BibSyntaxError("unterminated entry")
        if text[position] == closer:
            return entry, position + 1
        field = _FIELD_NAME_RE.match(text, position)
        if field is None:
            raise _BibSyntaxError(f"bad field in {entry['ID']}")
        value, position = _read_value(text, field.end(), strings)
        entry.setdefault(field.group(1).lower(), value)


_NEXT_ENTRY_RE = re.compile(r"\n[ \t]*@")


def fast_bib_parse(text: str, malformed: list[str] | None = None) -> list[dict[str, str]]:
    """Parse BibTeX into bibtexparser-style dicts (``ID``, ``ENTRYTYPE``, fields).

    Supports ``@string`` macros (including the month abbreviations), ``#``
    concatenation, braced and quoted values with nested braces, and entries
    delimited by ``( )``. An ``@`` inside a field value never starts an entry,
    and an entry on a line starting with ``%`` is treated as commented out.
    A malformed entry is skipped up to the next line starting with ``@``; its
    text is appended to ``malformed`` when a list is given.
    """

    strings = dict(_MONTH_MACROS)
    entries: list[dict[str, str]] = []
    position = 0
    while True:
        at = _AT_RE.search(text, position)
        if at is None:
            return entries
        line_start = text.rfind("\n", 0, at.start()) + 1
        if text[line_start : at.start()].lstrip().startswith("%"):
            position = at.end()
            continue
        try:
            record, position = _parse_entry(text, at, strings)
        except _BibSyntaxError:
            following = _NEXT_ENTRY_RE.search(text, at.end())
            position = following.start() + 1 if following else len(text)
            if malformed is not None:
                malformed.append(text[at.start() : position])
            continue
        if record is not None:
            entries.append(record)


def parse_bib_text(text: str) -> list[dict[str, str]]:
    """Parse BibTeX with the fast parser; retry malformed entries with bibtexparser.

    bibtexparser 1.x (pyparsing) parses only ~40 KB/s, which dominated the M2
    running time (on 150 corpus files: 2125 s versus 6 s, identical records
    except duplicated keys, where the fast parser keeps the first occurrence
    like BibTeX does). bibtexparser is still used for the few entries the fast
    parser rejects because it recovers some malformed input.
    """

    malformed: list[str] = []
    entries = fast_bib_parse(text, malformed)
    for chunk in malformed:
        try:
            entries.extend(_bibtexparser_loads(chunk))
        except Exception as error:  # bibtexparser raises several exception types
            LOGGER.debug("Skipping malformed BibTeX entry (%s): %.60s", error, chunk)
    return entries


def _bib_record_to_entry(record: dict[str, str], arxiv_id: str | None) -> Entry:
    fields = {key.lower(): value for key, value in record.items()}
    eprint = None
    if fields.get("archiveprefix", "").lower() == "arxiv" or "eprint" in fields:
        eprint = normalize_arxiv(fields.get("eprint"))
    for candidate in ("arxivid", "arxiv", "journal", "url", "note", "howpublished", "volume"):
        if eprint:
            break
        eprint = normalize_arxiv(fields.get(candidate))
    url = fields.get("url") or None
    if not url:
        match = _URL_RE.search(fields.get("howpublished", "") + " " + fields.get("note", ""))
        if match:
            url = next(group for group in match.groups() if group)
    doi = normalize_doi(fields.get("doi")) or normalize_doi(url)
    journal = fields.get("journal") or fields.get("journaltitle") or fields.get("booktitle")
    return Entry(
        citation_key=record.get("ID", "").strip(),
        entry_type=(record.get("ENTRYTYPE") or "misc").lower(),
        title=_clean_field(fields.get("title")),
        authors=parse_bibtex_authors(fields.get("author") or fields.get("editor")),
        year=parse_year(fields.get("year") or fields.get("date")),
        doi=doi,
        journal=_clean_field(journal),
        url=url.strip() if url else None,
        eprint=eprint,
        source="bib",
        arxiv_id=arxiv_id,
    )


def _resolve_bib_files(version_dir: Path, text: str) -> list[Path]:
    all_bibs = sorted(path for path in version_dir.rglob("*") if path.suffix.lower() == ".bib" and path.is_file())
    if not all_bibs:
        return []
    named: list[str] = []
    for match in _BIBLIOGRAPHY_RE.finditer(text):
        named.extend(name.strip() for name in match.group(1).split(",") if name.strip())
    resolved: list[Path] = []
    for name in named:
        name = name if name.lower().endswith(".bib") else name + ".bib"
        candidate = version_dir / name
        if candidate.is_file():
            resolved.append(candidate)
            continue
        matches = [path for path in all_bibs if path.name.lower() == Path(name).name.lower()]
        resolved.extend(matches[:1])
    # The named file is missing (often only a .bbl was shipped): fall back to
    # every .bib present. Entries are still filtered to the cited keys.
    return list(dict.fromkeys(resolved)) or all_bibs


# ---------------------------------------------------------------------------
# \bibitem conversion
# ---------------------------------------------------------------------------

_BIBITEM_RE = re.compile(r"\\bibitem(?![A-Za-z@])\s*")
_THEBIBLIOGRAPHY_END_RE = re.compile(r"\\end\s*\{thebibliography\}|\\end\s*\{document\}")
_BIBINFO_RE = re.compile(r"\\bibinfo\s*\{(\w+)\}\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
_ITALIC_RE = re.compile(
    r"\\(?:emph|textit|textsl)\s*\{((?:[^{}]|\{[^{}]*\})*)\}|\{\\(?:em|it|sl)\s+((?:[^{}]|\{[^{}]*\})*)\}"
)
_QUOTED_TITLE_RE = re.compile(r"``(.+?)''|\"(.+?)\"|“(.+?)”", re.DOTALL)
_PAREN_YEAR_RE = re.compile(r"\(\s*((?:19|20)\d{2})[a-z]?\s*\)")
# "Phys. Rev. Lett. \textbf{106}": capitalized words, each ended by "." or a
# space. Requiring the separator keeps the split unique; the earlier form
# ``(?:[A-Z][A-Za-z]*\.?\s*)+`` backtracked exponentially on acronyms such as
# "GAFA" and hung the runner (paper 2606-00335).
_ABBREVIATED_VENUE_RE = re.compile(r"(?:[A-Z][A-Za-z]*(?:\.\s*|\s+))+(?:\\textbf|\d)")
_ACCESS_NOTE_RE = re.compile(r"\[?\s*(?:last\s+)?(?:accessed|visited|retrieved)\b[^\]\n]*\]?", re.IGNORECASE)
_PROCEEDINGS_RE = re.compile(r"^\s*in\b|proc\.|proceedings|conference|workshop|symposium|\bconf\.", re.IGNORECASE)
_BOOK_RE = re.compile(
    r"\bpress\b|publish|springer|wiley|elsevier|cambridge|oxford|\bhall\b|verlag|\bed\.|edition",
    re.IGNORECASE,
)
_THESIS_RE = re.compile(r"\bthesis\b|dissertation", re.IGNORECASE)
_REPORT_RE = re.compile(r"technical report|tech\. rep|\breport\b", re.IGNORECASE)


def _read_group(text: str, start: int, opener: str, closer: str) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != opener:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def iter_bibitems(text: str) -> list[tuple[str, str, str]]:
    """Return ``(key, label, body)`` for every ``\\bibitem`` in ``text``."""

    items: list[tuple[str, str, str]] = []
    starts = [match for match in _BIBITEM_RE.finditer(text)]
    for index, match in enumerate(starts):
        position = match.end()
        label = ""
        optional = _read_group(text, position, "[", "]")
        if optional:
            label, position = optional
            while position < len(text) and text[position].isspace():
                position += 1
        key_group = _read_group(text, position, "{", "}")
        if key_group is None:
            continue
        key, position = key_group
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        stop = _THEBIBLIOGRAPHY_END_RE.search(text, position, end)
        body = text[position : stop.start() if stop else end]
        items.append((key.strip(), label, body))
    return items


def _venue_type(venue: str | None, body: str, eprint: str | None) -> str:
    venue = venue or ""
    if _THESIS_RE.search(body):
        return "phdthesis"
    if _PROCEEDINGS_RE.search(venue):
        return "inproceedings"
    if venue and not re.search(r"arxiv", venue, re.IGNORECASE):
        if _BOOK_RE.search(venue) and not re.search(r"\d+\s*\(\d+\)|vol\.|\\textbf", body):
            return "book"
        return "article"
    if _REPORT_RE.search(body):
        return "techreport"
    if _BOOK_RE.search(body) and not eprint:
        return "book"
    return "misc"


def _top_level_period_cut(text: str, start: int = 0) -> int:
    """Index of the first sentence-ending ``. `` at brace depth 0 (not an initial)."""

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char in ".?!" and depth == 0 and (index + 1 == len(text) or text[index + 1].isspace()):
            word = re.search(r"([A-Za-z]+)$", text[start:index])
            if char == "." and word and len(word.group(1)) <= 1:
                continue  # initial such as "A."
            return index
    return -1


def _authors_end_surname_first(text: str) -> int:
    """For "Smith, A., Jones, B. Title ..." return the end of the author list."""

    for match in re.finditer(r"(?<![A-Za-z])[A-Z][a-z]?\.(?:\s*-?[A-Z][a-z]?\.)*\s+(?=[A-Z0-9\\{“\"`])", text):
        rest = text[match.end() :]
        if re.match(r"(?:and\b|&|\\&|[A-Z][a-z]?\.|[A-Z][\w'\-]+,\s*[A-Z][a-z]?\.)", rest):
            continue
        return match.end()
    return -1


def _parse_bibitem_body(key: str, label: str, body: str, arxiv_id: str | None, source: str) -> Entry:
    body = _DROPPED_COMMANDS_RE.sub(lambda match: " " if "newblock" not in match.group(0) else match.group(0), body)
    body = body.strip()
    urls = [next(group for group in match.groups() if group) for match in _URL_RE.finditer(body)]
    doi = normalize_doi(body)
    eprint = normalize_arxiv(body)
    url = next((value for value in urls if "doi.org" not in value), None) or (urls[0] if urls else None)
    # Years inside identifiers (DOIs, arXiv IDs, URLs) or access notes are not
    # publication years.
    year_text = _URL_RE.sub(" ", _DOI_RE.sub(" ", body))
    year_text = _ACCESS_NOTE_RE.sub(" ", year_text)
    year_text = re.sub(r"\d{4}\.\d{4,5}", " ", year_text)
    paren_year = _PAREN_YEAR_RE.search(label) or _PAREN_YEAR_RE.search(year_text)
    years = _YEAR_RE.findall(year_text)
    year = int(paren_year.group(1)) if paren_year else (int(years[-1]) if years else None)

    authors_text = title = venue = None
    info = defaultdict(list)
    for name, value in _BIBINFO_RE.findall(body):
        info[name.lower()].append(value)
    if info:
        authors = [latex_to_text(value) for value in info.get("author", [])]
        title = info.get("title", [None])[0]
        venue = (info.get("journal") or info.get("booktitle") or [None])[0]
        year = parse_year((info.get("year") or [None])[0]) or year
        doi = normalize_doi((info.get("doi") or [None])[0]) or doi
        eprint = normalize_arxiv((info.get("eprint") or [None])[0]) or eprint
        return Entry(
            citation_key=key,
            entry_type=_venue_type(venue, body, eprint),
            title=_clean_field(title),
            authors=[name for name in authors if name],
            year=year,
            doi=doi,
            journal=_clean_field(venue),
            url=url,
            eprint=eprint,
            source=source,
            arxiv_id=arxiv_id,
        )

    if "\\newblock" in body:
        blocks = [block.strip() for block in re.split(r"\\newblock", body) if block.strip()]
        authors_text = _PAREN_YEAR_RE.sub("", blocks[0]) if blocks else ""
        title = blocks[1] if len(blocks) > 1 else None
        venue_block = blocks[2] if len(blocks) > 2 else None
        if venue_block:
            italic = _ITALIC_RE.search(venue_block)
            venue = (italic.group(1) or italic.group(2)) if italic else venue_block.split(",")[0]
    else:
        quoted = _QUOTED_TITLE_RE.search(body)
        if quoted:
            authors_text = body[: quoted.start()]
            title = next(group for group in quoted.groups() if group)
            rest = body[quoted.end() :]
            italic = _ITALIC_RE.search(rest)
            if italic:
                venue = italic.group(1) or italic.group(2)
            else:
                # "Phys. Rev. D \textbf{34}, 470": the venue ends at the volume.
                venue = re.split(r"\\textbf|,|\s\d", rest.strip(" ,."))[0]
        else:
            paren = _PAREN_YEAR_RE.search(body)
            first_italic = _ITALIC_RE.search(body)
            first_bold = body.find("\\textbf")
            if (
                paren
                and paren.start() < 400
                and (first_italic is None or paren.start() < first_italic.start())
                and (first_bold < 0 or paren.start() < first_bold)
            ):
                # APA/author-year: "Authors (2020). Title. Venue, ..."
                authors_text = body[: paren.start()]
                rest = body[paren.end() :].lstrip(" .,")
                italic = _ITALIC_RE.match(rest)
                if italic:
                    title = italic.group(1) or italic.group(2)
                    rest = rest[italic.end() :]
                else:
                    cut = _top_level_period_cut(rest)
                    title, rest = (rest[:cut], rest[cut + 1 :]) if cut > 0 else (rest, "")
                italic = _ITALIC_RE.search(rest)
                venue = (italic.group(1) or italic.group(2)) if italic else rest.strip(" ,.").split(",")[0]
            else:
                italic = _ITALIC_RE.search(body)
                surname_first = re.match(r"\s*[A-Z][^\s,]+(?:\s[A-Z][^\s,]+)?,\s*[A-Z][a-z]?\.", latex_to_text(body))
                if surname_first:
                    end = _authors_end_surname_first(body)
                    authors_text = body[:end] if end > 0 else body.split(".")[0]
                    rest = body[end:] if end > 0 else ""
                    cut = _top_level_period_cut(rest)
                    if italic and (cut < 0 or italic.start() - end <= cut):
                        title = rest[: max(0, italic.start() - end)].strip(" ,.") or None
                    else:
                        title = rest[:cut] if cut > 0 else None
                    venue = (italic.group(1) or italic.group(2)) if italic else None
                else:
                    # "I. Surname, J. Other, and K. Last, Title, Venue ..." (math
                    # and physics styles). Physics often omits the title.
                    and_match = None
                    for and_match in re.finditer(r"\band\b|\\&", body[: italic.start() if italic else len(body)]):
                        pass
                    search_from = and_match.end() if and_match else 0
                    comma = re.search(r",", body[search_from:])
                    end = search_from + comma.start() if comma else -1
                    if italic and (end < 0 or italic.start() < end):
                        end = italic.start()
                    authors_text = body[:end] if end > 0 else ""
                    rest = body[end:].lstrip(" ,") if end > 0 else body
                    if italic and italic.start() >= end:
                        before = body[end : italic.start()].strip(" ,.")
                        if before and len(before.split()) >= 3:
                            title, venue = before, italic.group(1) or italic.group(2)
                        else:
                            title = italic.group(1) or italic.group(2)
                            after = body[italic.end() :].strip(" ,.")
                            venue = re.split(r"\\textbf|\(|,|\d", after)[0] or None
                    elif _ABBREVIATED_VENUE_RE.match(rest):
                        venue = re.split(r"\\textbf|\d", rest)[0]
                    else:
                        cut = _top_level_period_cut(rest)
                        title = rest[:cut] if cut > 0 else None

    venue_text = _clean_field(venue)
    entry_type = _venue_type(venue_text, body, eprint)
    if venue_text:
        venue_text = re.sub(r"^in:?\s+", "", venue_text, flags=re.IGNORECASE) or None
    return Entry(
        citation_key=key,
        entry_type=entry_type,
        title=_clean_field(title),
        authors=parse_free_authors(authors_text or ""),
        year=year,
        doi=doi,
        journal=venue_text,
        url=url,
        eprint=eprint,
        source=source,
        arxiv_id=arxiv_id,
    )


def parse_bibitems(text: str, arxiv_id: str | None = None, source: str = "bibitem") -> list[Entry]:
    entries: list[Entry] = []
    for key, label, body in iter_bibitems(text):
        if not key:
            continue
        try:
            entries.append(_parse_bibitem_body(key, label, body, arxiv_id, source))
        except Exception as error:  # one malformed item must not drop the bibliography
            LOGGER.debug("Cannot parse bibitem %s: %s", key, error)
            entries.append(Entry(citation_key=key, source=source, arxiv_id=arxiv_id))
    return entries


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------


def _publication_id(version_dir: Path) -> str:
    return version_dir.parent.name.strip().replace(".", "-")


def extract(version_dir: str | Path, text: str) -> list[Entry]:
    """Return the bibliography of one version.

    ``text`` is the flattened compilation text from ``gather.flatten``. Entries
    from ``.bib`` files are restricted to keys that the text cites (unless it
    uses ``\\nocite{*}``) because shared ``.bib`` files often contain hundreds
    of unrelated entries.
    """

    version_dir = Path(version_dir)
    arxiv_id = _publication_id(version_dir)
    entries = parse_bibitems(text, arxiv_id, "bibitem")
    seen = {entry.citation_key for entry in entries}

    if _BIBLIOGRAPHY_RE.search(text):
        cited = set(cite_keys(text))
        keep_all = bool(_NOCITE_ALL_RE.search(text))
        bib_files = _resolve_bib_files(version_dir, text)
        for path in bib_files:
            raw, _ = read_tex(path)
            for record in parse_bib_text(select_entries(raw, None if keep_all else cited - seen)):
                key = record.get("ID", "").strip()
                if not key or key in seen or (not keep_all and key not in cited):
                    continue
                entries.append(_bib_record_to_entry(record, arxiv_id))
                seen.add(key)
        if not bib_files:
            # Only reachable when a group keeps .bbl files (not the case for M1).
            for path in sorted(version_dir.rglob("*.bbl")):
                raw, _ = read_tex(path)
                for entry in parse_bibitems(raw, arxiv_id, "bbl"):
                    if entry.citation_key not in seen:
                        entries.append(entry)
                        seen.add(entry.citation_key)
    return entries


def source_counts(entries_by_version: dict[Any, list[Entry]]) -> dict[str, int]:
    """Count entries before deduplication by origin (``bib``/``bibitem``/``bbl``)."""

    counts: Counter[str] = Counter()
    for entries in entries_by_version.values():
        counts.update(entry.source for entry in entries)
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _first_author_family(entry: Entry) -> str:
    if not entry.authors:
        return ""
    tokens = normalize_title(entry.authors[0]).split()
    return tokens[-1] if tokens else ""


def _token_similarity(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a and b else 0.0


class _Groups:
    """Union-find over entries that refuses merges with conflicting identifiers."""

    def __init__(self, entries: list[Entry]) -> None:
        self.parent = list(range(len(entries)))
        self.dois = [{entry.doi} if entry.doi else set() for entry in entries]
        self.arxivs = [{entry.eprint} if entry.eprint else set() for entry in entries]
        self.years = [{entry.year} if entry.year else set() for entry in entries]
        self.authors = [{_first_author_family(entry)} - {""} for entry in entries]
        self.titles = [{normalize_title(entry.title)} - {""} for entry in entries]

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def conflict(self, left: int, right: int, strict_titles: bool) -> bool:
        if self.dois[left] and self.dois[right] and not self.dois[left] & self.dois[right]:
            return True
        if self.arxivs[left] and self.arxivs[right] and not self.arxivs[left] & self.arxivs[right]:
            return True
        if strict_titles:
            if self.years[left] and self.years[right] and min(
                abs(a - b) for a in self.years[left] for b in self.years[right]
            ) > 1:
                return True
            if self.authors[left] and self.authors[right] and not self.authors[left] & self.authors[right]:
                return True
        elif self.titles[left] and self.titles[right]:
            best = max(_token_similarity(a, b) for a in self.titles[left] for b in self.titles[right])
            if best < 0.5:
                return True
        return False

    def union(self, left: int, right: int, *, force: bool = False, strict_titles: bool = False) -> None:
        left, right = self.find(left), self.find(right)
        if left == right or (not force and self.conflict(left, right, strict_titles)):
            return
        self.parent[right] = left
        for attribute in (self.dois, self.arxivs, self.years, self.authors, self.titles):
            attribute[left] |= attribute[right]


def _merge_group(members: list[tuple[int, Entry]], key: str) -> dict[str, Any]:
    # Most recent version first; inside a version prefer .bib and fuller entries.
    ordered = [
        entry
        for _, entry in sorted(
            members, key=lambda item: (-item[0], item[1].source != "bib", -item[1].completeness())
        )
    ]
    merged: dict[str, Any] = {name: next((getattr(e, name) for e in ordered if getattr(e, name)), None) for name in MERGED_FIELDS}
    authors = next((entry.authors for entry in ordered if entry.authors), [])
    entry_type = next((entry.entry_type for entry in ordered if entry.entry_type != "misc"), "misc")
    url = merged["url"] or (f"https://arxiv.org/abs/{merged['eprint']}" if merged["eprint"] else None)
    return {
        "arxiv_id": ordered[0].arxiv_id,
        "citation_key": key,
        "entry_type": entry_type,
        "title": merged["title"],
        "authors": list(authors),
        "year": merged["year"],
        "doi": merged["doi"],
        "journal": merged["journal"],
        "url": url,
    }


def dedup(entries_by_version: dict[Any, list[Entry]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, str]]]:
    """Merge duplicate references within and across versions.

    Two entries describe the same work when they share a DOI or an arXiv ID,
    when their normalized titles match (and neither year nor first author
    contradicts it), or when they use the same citation key in different
    versions without contradicting identifiers. The chosen key of a group is the
    one used by its most recent version, so ``vaswani2017`` (v1) and
    ``vit_transformer`` (v2, v3) become ``vit_transformer``. Fields are unioned,
    preferring values from more recent versions.

    Returns the unified rows (columns of the ``bibtex`` table) and
    ``{version: {old_key: chosen_key}}`` for every version.
    """

    flat: list[tuple[int, Entry]] = []
    for version, entries in entries_by_version.items():
        number = int(str(version).lower().removeprefix("v"))
        flat.extend((number, entry) for entry in entries if entry.citation_key)
    rename_maps: dict[int, dict[str, str]] = {
        int(str(version).lower().removeprefix("v")): {} for version in entries_by_version
    }
    if not flat:
        return [], rename_maps

    groups = _Groups([entry for _, entry in flat])
    first_by_version_key: dict[tuple[int, str], int] = {}
    by_doi: dict[str, int] = {}
    by_arxiv: dict[str, int] = {}
    by_title: dict[str, list[int]] = defaultdict(list)
    by_key: dict[str, list[int]] = defaultdict(list)

    for index, (version, entry) in enumerate(flat):
        # 1. The same key inside one version is one entry (e.g. two .bib files).
        previous = first_by_version_key.setdefault((version, entry.citation_key), index)
        if previous != index:
            groups.union(previous, index, force=True)
    for index, (version, entry) in enumerate(flat):
        # 2. Persistent identifiers.
        if entry.doi:
            groups.union(by_doi.setdefault(entry.doi, index), index)
        if entry.eprint:
            groups.union(by_arxiv.setdefault(entry.eprint, index), index)
    for index, (version, entry) in enumerate(flat):
        # 3. Identical normalized titles (long enough to be distinctive).
        title = normalize_title(entry.title)
        if len(title) >= 12 and len(title.split()) >= 2:
            for other in by_title[title]:
                groups.union(other, index, strict_titles=True)
            by_title[title].append(index)
    for index, (version, entry) in enumerate(flat):
        # 4. A key kept across versions refers to the same work unless the
        #    identifiers or the titles disagree.
        for other in by_key[entry.citation_key]:
            if flat[other][0] != version:
                groups.union(other, index)
        by_key[entry.citation_key].append(index)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(flat)):
        members[groups.find(index)].append(index)

    # Choose keys: the most recent version's key, then the most frequent one.
    taken: set[str] = set()
    chosen: dict[int, str] = {}
    ordered_groups = sorted(
        members.items(), key=lambda item: (-max(flat[i][0] for i in item[1]), min(item[1]))
    )
    for root, indices in ordered_groups:
        latest = max(flat[i][0] for i in indices)
        frequency = Counter(flat[i][1].citation_key for i in indices)
        in_latest = {flat[i][1].citation_key for i in indices if flat[i][0] == latest}
        candidates = sorted(frequency, key=lambda key: (key not in in_latest, -frequency[key], key))
        key = next((candidate for candidate in candidates if candidate not in taken), None)
        if key is None:
            base, suffix = candidates[0], 2
            while f"{base}_{suffix}" in taken:
                suffix += 1
            key = f"{base}_{suffix}"
        taken.add(key)
        chosen[root] = key

    unified: list[dict[str, Any]] = []
    for root, indices in ordered_groups:
        key = chosen[root]
        for index in indices:
            version, entry = flat[index]
            if entry.citation_key != key:
                rename_maps.setdefault(version, {})[entry.citation_key] = key
        unified.append(_merge_group([flat[index] for index in indices], key))
    unified.sort(key=lambda row: row["citation_key"])
    return unified, rename_maps
