"""Pure ``str -> str`` text standardization used after the hierarchy is built.

``run_m2.py`` applies the functions to every node in the order
``rename_cites -> normalize_math -> clean``, so cleaned content is what the
store hashes for cross-version deduplication. ``split_sentences`` runs earlier,
inside ``hierarchy.build_tree``, on raw (uncleaned) LaTeX prose.

References used while writing this module:

* LaTeX math modes: L. Lamport, *LaTeX: A Document Preparation System*, 2nd ed.,
  sec. 3.3, and the ``amsmath`` user's guide (https://ctan.org/pkg/amsmath).
* natbib/biblatex citation commands: https://ctan.org/pkg/natbib and
  https://ctan.org/pkg/biblatex.
* Abbreviation-aware sentence splitting follows the idea of Punkt
  (T. Kiss and J. Strunk, Computational Linguistics 32(4), 2006), reduced to a
  fixed abbreviation list because the corpus is English scientific prose.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .gather import strip_comments

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Lower-case abbreviations (without the final period) that never end a sentence
# when followed by more text. "etc" is deliberately absent: it usually does.
ABBREVIATIONS = {
    "e.g", "i.e", "ie", "cf", "c.f", "vs", "viz", "resp", "approx", "w.r.t", "a.k.a",
    "et al", "al", "fig", "figs", "eq", "eqs", "eqn", "eqns", "sec", "secs", "sect",
    "tbl", "refs", "thm", "thms", "lem", "lems", "defn", "props", "algo", "appx",
    "chap", "eds", "dr", "mr", "mrs", "ms", "prof", "jr", "sr", "dept", "univ", "ibid",
    "jan", "feb", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
# Abbreviations that are also ordinary words ("a mobile app.", "the answer is no.")
# and are only treated as abbreviations before a number or a reference.
NUMBERED_ABBREVIATIONS = {
    "no", "nos", "p", "pp", "vol", "vols", "ch", "pt", "ex", "def", "rem", "prop",
    "cor", "ref", "tab", "alg", "app", "art", "obs", "ass", "conj", "claim", "step",
}

# Inline formatting whose argument is ordinary running text: a sentence may end
# inside it (``\textbf{Contributions.} We ...``).
TEXT_FORMAT_COMMANDS = {
    "emph", "textbf", "textit", "textsl", "textsc", "texttt", "textrm", "textsf",
    "textup", "textmd", "textnormal", "underline", "uline",
}
# Commands whose argument may contain a period that ends nothing in the prose,
# for example ``end of sentence.\footnote{Text. More text.} Next``.
TRAILING_COMMANDS = {
    "footnote", "footnotemark", "footnotetext", "label", "ref", "eqref", "cref", "Cref",
    "autoref", "url", "href", "thanks",
}
MATH_ENVIRONMENTS = {
    "math", "displaymath", "equation", "equation*", "align", "align*", "alignat",
    "alignat*", "flalign", "flalign*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*",
}

_COMMAND_RE = re.compile(r"\\([A-Za-z@]+)\*?")
_ENV_NAME_RE = re.compile(r"\s*\{([^{}]+)\}")
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s{}]*[^\s{}.,;:!?)\]'\"]")
_TERMINATOR_RE = re.compile(r"[.!?]+")
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n\s*")
_CLOSERS = "\"')]}’”"
_WHITESPACE_RE = re.compile(r"\s+")


def _skip_balanced(text: str, start: int, opener: str, closer: str) -> int:
    """Return the index after the group opened at ``start`` (or ``len(text)``)."""

    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def _find_unescaped(text: str, needle: str, start: int) -> int:
    index = text.find(needle, start)
    while index >= 0:
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return index
        index = text.find(needle, index + 1)
    return -1


def _protected_spans(text: str) -> tuple[bytearray, dict[int, int]]:
    """Mark characters where a sentence may not end.

    Returns a byte mask (1 = protected) and, for trailing commands such as
    ``\\footnote``, a map from the command start to the end of its arguments.
    """

    length = len(text)
    mask = bytearray(length)
    command_ends: dict[int, int] = {}

    def protect(start: int, end: int) -> None:
        end = min(end, length)
        mask[start:end] = b"\x01" * (end - start)

    for url in _URL_RE.finditer(text):
        protect(url.start(), url.end())

    index = 0
    while index < length:
        char = text[index]
        if char == "$":
            if text.startswith("$$", index):
                close = _find_unescaped(text, "$$", index + 2)
                end = length if close < 0 else close + 2
            else:
                close = _find_unescaped(text, "$", index + 1)
                end = length if close < 0 else close + 1
            protect(index, end)
            index = end
            continue
        if char != "\\" or index + 1 >= length:
            index += 1
            continue
        following = text[index + 1]
        if following in "([":
            close = _find_unescaped(text, "\\)" if following == "(" else "\\]", index + 2)
            end = length if close < 0 else close + 2
            protect(index, end)
            index = end
            continue
        command = _COMMAND_RE.match(text, index)
        if command is None:
            index += 2  # escaped character such as \. \% \$
            continue
        name = command.group(1)
        end = command.end()
        if name == "begin":
            environment = _ENV_NAME_RE.match(text, end)
            if environment and environment.group(1).strip() in MATH_ENVIRONMENTS:
                env = re.escape(environment.group(1).strip())
                close = re.compile(r"\\end\s*\{" + env + r"\}").search(text, environment.end())
                end = length if close is None else close.end()
                protect(index, end)
                index = end
                continue
            index = environment.end() if environment else end
            continue
        if name in TEXT_FORMAT_COMMANDS:
            index = end
            continue
        # Protect the command's own optional and mandatory arguments.
        cursor = end
        while cursor < length:
            probe = cursor
            while probe < length and text[probe] in " \t":
                probe += 1
            if probe < length and text[probe] == "[" and name not in {"item"}:
                cursor = _skip_balanced(text, probe, "[", "]")
            elif probe < length and text[probe] == "{":
                cursor = _skip_balanced(text, probe, "{", "}")
            else:
                break
        protect(index, cursor)
        if name in TRAILING_COMMANDS or "cite" in name.lower():
            command_ends[index] = cursor
        index = max(cursor, index + 1)
    return mask, command_ends


def _word_before(text: str, index: int) -> str:
    """Return the token ending at ``index`` (letters and inner periods)."""

    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    word = text[start:index].lower()
    # "et al." is written as two tokens.
    if word == "al" and text[max(0, start - 3) : start].lower().strip() == "et":
        return "et al"
    return word


def _is_boundary(text: str, mask: bytearray, command_ends: dict[int, int], match: re.Match[str]) -> int:
    """Return the cut position after the terminator ``match`` or -1."""

    start, end = match.start(), match.end()
    if mask[start]:
        return -1
    length = len(text)
    # Closing quotes, brackets and trailing \footnote{...}/\cite{...} commands
    # belong to the sentence that just ended.
    while end < length:
        if text[end] in _CLOSERS or text.startswith("''", end):
            end += 1
        elif end in command_ends:
            end = command_ends[end]
        else:
            break
    if end >= length:
        return length
    if not text[end].isspace():
        return -1
    following = end
    while following < length and text[following].isspace():
        following += 1
    if following >= length:
        return length
    nxt = text[following]
    if nxt.islower() or nxt in ",;:":
        return -1
    if text[start] == ".":
        word = _word_before(text, start)
        if word in ABBREVIATIONS:
            return -1
        if word in NUMBERED_ABBREVIATIONS and (nxt.isdigit() or nxt in "\\([~"):
            return -1
        # A single capital letter is an initial ("J. Smith") or an enumerator.
        if len(word) == 1 and text[start - 1].isupper():
            return -1
    return end


def split_sentences(text: str) -> list[str]:
    """Split LaTeX prose into sentences.

    No split happens inside math (``$..$``, ``\\(..\\)``, ``\\[..\\]``, math
    environments), inside command arguments such as ``\\cite{..}`` or
    ``\\footnote{..}``, inside URLs, after known abbreviations (``e.g.``,
    ``et al.``, ``Fig. 1``), after initials, or inside decimal numbers. A blank
    line (paragraph break) always ends a sentence.
    """

    if not text or not text.strip():
        return []
    mask, command_ends = _protected_spans(text)
    cuts = [0]
    for match in _TERMINATOR_RE.finditer(text):
        cut = _is_boundary(text, mask, command_ends, match)
        if cut > cuts[-1]:
            cuts.append(cut)
    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        if not mask[match.start()]:
            cuts.append(match.start())
    cuts = sorted(set(cuts + [len(text)]))
    sentences: list[str] = []
    for begin, end in zip(cuts, cuts[1:]):
        sentence = _WHITESPACE_RE.sub(" ", text[begin:end]).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


# ---------------------------------------------------------------------------
# Math normalization
# ---------------------------------------------------------------------------

BLOCK_MATH_ENVIRONMENTS = (
    "equation", "displaymath", "align", "alignat", "flalign", "gather", "multline", "eqnarray",
)
_BLOCK_BEGIN_RE = re.compile(
    r"\\begin\s*\{(?:" + "|".join(BLOCK_MATH_ENVIRONMENTS) + r")\*?\}(?:\s*\{\d+\})?\s*"
)
_BLOCK_END_RE = re.compile(r"\s*\\end\s*\{(?:" + "|".join(BLOCK_MATH_ENVIRONMENTS) + r")\*?\}")
_INLINE_ENV_RE = re.compile(r"\\begin\s*\{math\}\s*(.*?)\s*\\end\s*\{math\}", re.DOTALL)
# ``\\\\`` comes first so the line break ``\\[2pt]`` is never read as ``\[``.
_DELIMITER_RE = re.compile(r"\\\\|\\\(|\\\)|\\\[|\\\]")
_DOUBLE_DOLLAR_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)
_EQUATION_PADDING_RE = re.compile(r"(\\begin\{equation\})\s+|\s+(\\end\{equation\})")


def normalize_math(text: str) -> str:
    """Unify math delimiters.

    Inline math (``\\(..\\)``, ``\\begin{math}``) becomes ``$..$``. Display math
    (``$$..$$``, ``\\[..\\]``, ``equation*``, ``align(*)``, ``gather(*)``,
    ``multline(*)``, ``eqnarray(*)``, ``flalign(*)``, ``alignat(*)``,
    ``displaymath``) becomes ``\\begin{equation}..\\end{equation}``. Inner
    structures such as ``aligned``/``split``/``cases`` and the ``&``/``\\\\``
    alignment marks are kept unchanged.
    """

    if "\\" not in text and "$" not in text:
        return text
    text = _INLINE_ENV_RE.sub(lambda match: f"${match.group(1)}$", text)

    def delimiter(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in ("\\(", "\\)"):
            return "$"
        if token == "\\[":
            return "\\begin{equation}"
        if token == "\\]":
            return "\\end{equation}"
        return token

    text = _DELIMITER_RE.sub(delimiter, text)
    text = _DOUBLE_DOLLAR_RE.sub(lambda match: f"\\begin{{equation}}{match.group(1)}\\end{{equation}}", text)
    text = _BLOCK_BEGIN_RE.sub(lambda _: "\\begin{equation}", text)
    text = _BLOCK_END_RE.sub(lambda _: "\\end{equation}", text)
    return _EQUATION_PADDING_RE.sub(lambda match: match.group(1) or match.group(2), text)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Commands without arguments that only change layout or font.
FORMATTING_SWITCHES = (
    "centering", "raggedright", "raggedleft", "noindent", "indent", "par", "newpage",
    "clearpage", "cleardoublepage", "pagebreak", "linebreak", "nolinebreak", "allowbreak",
    "hline", "toprule", "midrule", "bottomrule", "hfill", "vfill", "hfil", "vfil",
    "smallskip", "medskip", "bigskip", "FloatBarrier", "maketitle", "protect", "sloppy",
    "fussy", "selectfont", "normalsize", "small", "footnotesize", "scriptsize", "tiny",
    "large", "Large", "LARGE", "huge", "Huge", "bfseries", "itshape", "slshape",
    "scshape", "upshape", "mdseries", "rmfamily", "sffamily", "ttfamily", "normalfont",
    "em", "bf", "it", "rm", "sf", "tt", "sc", "sl", "nonumber", "notag", "xspace",
    "frenchspacing", "nopagebreak", "endgraf", "strut", "unskip", "ignorespaces",
    "tabularnewline", "relax",
)
# Commands removed together with their arguments: (name, optional args, mandatory args).
REMOVED_WITH_ARGUMENTS = (
    ("label", 0, 1), ("vspace", 0, 1), ("hspace", 0, 1), ("addvspace", 0, 1),
    ("setlength", 0, 2), ("addtolength", 0, 2), ("cmidrule", 1, 1),
    ("cline", 0, 1), ("specialrule", 0, 3), ("rowcolor", 1, 1), ("cellcolor", 1, 1),
    ("arrayrulecolor", 1, 1), ("captionsetup", 1, 1), ("fontsize", 0, 2),
    ("linespread", 0, 1), ("pagestyle", 0, 1), ("thispagestyle", 0, 1),
    ("enlargethispage", 0, 1), ("addlinespace", 1, 0), ("needspace", 0, 1),
    ("rule", 1, 2), ("color", 1, 1), ("definecolor", 0, 3), ("phantomsection", 0, 0),
    ("renewcommand", 0, 2), ("setcounter", 0, 2), ("addtocounter", 0, 2),
)
# Commands replaced by their last mandatory argument: (name, optional, mandatory).
UNWRAPPED_COMMANDS = (
    ("textbf", 0, 1), ("textit", 0, 1), ("emph", 0, 1), ("textsl", 0, 1),
    ("texttt", 0, 1), ("textrm", 0, 1), ("textsf", 0, 1), ("textsc", 0, 1),
    ("textup", 0, 1), ("textmd", 0, 1), ("textnormal", 0, 1), ("underline", 0, 1),
    ("uline", 0, 1), ("mbox", 0, 1), ("resizebox", 0, 3), ("scalebox", 1, 2),
    ("adjustbox", 0, 2), ("textcolor", 1, 2),
)
FLOAT_ENVIRONMENTS = (
    "figure", "table", "wrapfigure", "wraptable", "sidewaysfigure", "sidewaystable",
    "longtable", "subfigure", "subtable", "algorithm",
)

_SWITCH_RE = re.compile(
    r"\\(?:" + "|".join(sorted(FORMATTING_SWITCHES, key=len, reverse=True)) + r")\*?(?![A-Za-z@])"
)
_FLOAT_OPTIONS_RE = re.compile(
    r"(\\begin\s*\{(?:" + "|".join(FLOAT_ENVIRONMENTS) + r")\*?\})"
    r"(?:\s*\[[^\[\]]*\])?"
)
_WRAP_ARGUMENTS_RE = re.compile(
    r"(\\begin\s*\{(?:wrapfigure|wraptable)\})\s*\{[^{}]*\}\s*\{(?:[^{}]|\{[^{}]*\})*\}"
)
# ``~`` is a non-breaking space, except inside URLs (``\url{http://x.edu/~me}``).
_TILDE_RE = re.compile(r"\\(?:url|href)\s*\{[^{}]*\}|(?:https?|ftp)://[^\s{}]+|(?<!\\)~")
_SPACING_RE = re.compile(r"\\[@/](?![A-Za-z])")


def _parse_arguments(text: str, start: int, optional: int, mandatory: int) -> tuple[list[str], int] | None:
    """Read up to ``optional`` ``[..]`` and exactly ``mandatory`` ``{..}`` groups."""

    index = start
    for _ in range(optional):
        probe = index
        while probe < len(text) and text[probe].isspace():
            probe += 1
        if probe < len(text) and text[probe] == "[":
            index = _skip_balanced(text, probe, "[", "]")
    if optional and text.startswith("(", index):
        # \cmidrule(lr){2-3}: trimming specification in parentheses.
        close = text.find(")", index)
        index = index if close < 0 else close + 1
    arguments: list[str] = []
    for _ in range(mandatory):
        probe = index
        while probe < len(text) and text[probe].isspace():
            probe += 1
        if probe >= len(text) or text[probe] != "{":
            return None
        end = _skip_balanced(text, probe, "{", "}")
        arguments.append(text[probe + 1 : end - 1])
        index = end
    return arguments, index


@lru_cache(maxsize=None)
def _command_pattern(specs: tuple[tuple[str, int, int], ...]) -> re.Pattern[str]:
    names = sorted((name for name, _, _ in specs), key=len, reverse=True)
    return re.compile(r"\\(" + "|".join(names) + r")\*?(?![A-Za-z@])")


def _rewrite_commands(text: str, specs: tuple[tuple[str, int, int], ...], keep_last: bool) -> str:
    names = {name: (optional, mandatory) for name, optional, mandatory in specs}
    pattern = _command_pattern(specs)
    if not pattern.search(text):
        return text
    output: list[str] = []
    position = 0
    while True:
        match = pattern.search(text, position)
        if match is None:
            output.append(text[position:])
            return "".join(output)
        optional, mandatory = names[match.group(1)]
        parsed = _parse_arguments(text, match.end(), optional, mandatory)
        output.append(text[position : match.start()])
        if parsed is None:
            # Malformed use: keep it unchanged rather than guess.
            output.append(match.group(0))
            position = match.end()
            continue
        arguments, end = parsed
        if keep_last and arguments:
            # Unwrapped arguments may themselves contain removable commands.
            output.append(_rewrite_commands(arguments[-1], specs, keep_last))
        position = end


def clean(text: str) -> str:
    """Remove formatting that has no semantic meaning and normalize spacing.

    Examples: ``\\centering``, ``[htpb]`` after ``\\begin{figure}``, booktabs
    rules (``\\toprule``/``\\midrule``/``\\bottomrule``), ``\\vspace{..}``,
    ``\\label{..}``, font size and font switches, ``\\textbf{x} -> x``,
    ``\\resizebox{..}{..}{x} -> x``, ``~ -> space``. Line breaks ``\\\\`` and
    alignment ``&`` are kept because they carry table/equation structure.
    """

    if not text:
        return ""
    if "%" in text:
        text = strip_comments(text)
    if "\\" in text:
        text = _FLOAT_OPTIONS_RE.sub(r"\1", text)
        text = _WRAP_ARGUMENTS_RE.sub(r"\1", text)
        text = _rewrite_commands(text, REMOVED_WITH_ARGUMENTS, keep_last=False)
        text = _rewrite_commands(text, UNWRAPPED_COMMANDS, keep_last=True)
        text = _SWITCH_RE.sub(" ", text)
        text = _SPACING_RE.sub("", text)
    if "~" in text:
        text = _TILDE_RE.sub(lambda match: " " if match.group(0) == "~" else match.group(0), text)
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Citation renaming
# ---------------------------------------------------------------------------

# Matches \cite, \citep, \citet, \citealp, \citeauthor, \citeyear, \Citet,
# \parencite, \textcite, \autocite, \footcite, \supercite, \nocite and the
# biblatex multi-cite forms (\cites, \parencites, ...).
_CITE_RE = re.compile(
    r"\\(?P<cmd>[A-Za-z]*cite[A-Za-z]*)\*?(?![A-Za-z@])"
    r"(?P<opts>(?:\s*\[[^\[\]]*\])*)"
    r"(?P<groups>(?:\s*\{[^{}]*\})+)"
)
_CITE_GROUP_RE = re.compile(r"(\s*)\{([^{}]*)\}")
_NOT_CITE_COMMANDS = {"citestyle", "citeindextrue", "citeindexfalse", "citetext"}


def cite_keys(text: str) -> list[str]:
    """Return every citation key used in ``text`` in order of appearance."""

    keys: list[str] = []
    for match in _CITE_RE.finditer(text):
        if match.group("cmd") in _NOT_CITE_COMMANDS:
            continue
        groups = _CITE_GROUP_RE.findall(match.group("groups"))
        if not match.group("cmd").endswith("s"):
            groups = groups[:1]
        for _, content in groups:
            keys.extend(key.strip() for key in content.split(",") if key.strip())
    return keys


def rename_cites(text: str, rename_map: dict[str, str]) -> str:
    """Rename keys inside citation commands according to ``rename_map``.

    ``\\cite{a, b,c}`` with ``{"a": "x", "c": "x"}`` becomes ``\\cite{x,b}``:
    keys are renamed, repeated keys are collapsed, and the separator spacing is
    normalized so cleanup-time deduplication is not affected by it.
    """

    if not text or "cite" not in text:
        return text

    def rename_group(content: str) -> str:
        seen: list[str] = []
        for key in content.split(","):
            key = key.strip()
            if not key:
                continue
            key = rename_map.get(key, key)
            if key not in seen:
                seen.append(key)
        return ",".join(seen)

    def replace(match: re.Match[str]) -> str:
        command = match.group("cmd")
        if command in _NOT_CITE_COMMANDS:
            return match.group(0)
        groups = match.group("groups")
        if command.endswith("s"):
            renamed = _CITE_GROUP_RE.sub(lambda group: "{" + rename_group(group.group(2)) + "}", groups)
        else:
            # Only the first group holds keys for single-cite commands.
            first = _CITE_GROUP_RE.match(groups)
            renamed = "{" + rename_group(first.group(2)) + "}" + groups[first.end() :]
        prefix = match.group(0)[: match.start("groups") - match.start()]
        return prefix + renamed

    return _CITE_RE.sub(replace, text)
