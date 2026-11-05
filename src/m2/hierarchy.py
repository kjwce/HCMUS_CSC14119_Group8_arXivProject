"""Build a document hierarchy from a flattened LaTeX compilation path.

This module owns structural parsing only. Text cleanup, math normalization and
sentence-boundary quality remain the responsibility of ``normalize.py``. The
provided ``split_sentences`` callback keeps that boundary explicit.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import Node

SentenceSplitter = Callable[[str], list[str]]

HEADING_ORDER = (
    "part",
    "chapter",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "subparagraph",
)
LIST_ENVIRONMENTS = {"itemize", "enumerate", "description", "highlights"}
ABSTRACT_ENVIRONMENTS = {"abstract", "abstract*"}
KEYWORD_ENVIRONMENTS = {"keyword", "keywords"}
MATH_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
}
FIGURE_ENVIRONMENTS = {
    "figure",
    "figure*",
    "table",
    "table*",
    # Common standalone table environments.  They must also be recognized when
    # authors omit the outer ``table`` float (notably AASTeX ``deluxetable``).
    "tabular",
    "tabular*",
    "tabularx",
    "tabulary",
    "tblr",
    "longtblr",
    "talltblr",
    "deluxetable",
    "deluxetable*",
    "supertabular",
    "xtabular",
    "threeparttable",
    "ruledtabular",
    "wrapfigure",
    "wraptable",
    "longtable",
    "sidewaysfigure",
    "sidewaysfigure*",
    "sidewaystable",
    "sidewaystable*",
    "subfigure",
    "subtable",
}
SPECIAL_ENVIRONMENTS = (
    LIST_ENVIRONMENTS
    | ABSTRACT_ENVIRONMENTS
    | KEYWORD_ENVIRONMENTS
    | MATH_ENVIRONMENTS
    | FIGURE_ENVIRONMENTS
    | {"thebibliography"}
)

_HEADING_COMMAND_RE = re.compile(
    r"\\(?P<name>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
    r"(?P<star>\*)?\s*",
    re.IGNORECASE,
)
_SPECIAL_BEGIN_RE = re.compile(
    r"\\begin\s*\{(?P<env>"
    + "|".join(re.escape(name) for name in sorted(SPECIAL_ENVIRONMENTS, key=len, reverse=True))
    + r")\}",
    re.IGNORECASE,
)
_EVENT_RE = re.compile(
    r"\\(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*"
    r"|\\begin\s*\{(?:"
    + "|".join(re.escape(name) for name in sorted(SPECIAL_ENVIRONMENTS, key=len, reverse=True))
    + r")\}"
    r"|\$\$|\\\[|\\abstract\b|\\appendix\b|\\(?:bibliography|printbibliography)\b",
    re.IGNORECASE,
)
_BEGIN_END_RE = re.compile(r"\\(?P<action>begin|end)\s*\{(?P<env>[^{}]+)\}", re.IGNORECASE)
_ITEM_RE = re.compile(r"\\item(?:\s*\[[^\]]*\])?", re.IGNORECASE)

# Front-matter commands carry publication metadata rather than paper prose.  They
# occur both before and after ``\\begin{document}`` depending on the document
# class, so merely slicing out the preamble is insufficient.
_FRONT_MATTER_ARGUMENTS = {
    "@ifundefined": 3,
    "accepted": 1,
    "address": 1,
    "affil": 1,
    "affiliation": 1,
    "altaffiliation": 1,
    "articletype": 1,
    "author": 1,
    "authorrunning": 1,
    "collaboration": 1,
    "cormark": 0,
    "cortext": 1,
    "correspondingauthor": 1,
    "date": 1,
    "ead": 1,
    "email": 1,
    "fnmark": 0,
    "fntext": 1,
    "institute": 1,
    "keywords": 1,
    "makeatletter": 0,
    "makeatother": 0,
    "maketitle": 0,
    "received": 1,
    "revised": 1,
    "shortauthors": 1,
    "shorttitle": 1,
    "subject": 1,
    "submitjournal": 1,
    "tnotemark": 0,
    "tnotetext": 1,
    "thanks": 1,
    "title": 1,
    "titlerunning": 1,
}
_FRONT_MATTER_RE = re.compile(
    r"\\(?P<name>"
    + "|".join(re.escape(name) for name in sorted(_FRONT_MATTER_ARGUMENTS, key=len, reverse=True))
    + r")\*?(?![A-Za-z@])",
    re.IGNORECASE,
)
_RESIDUAL_ENVIRONMENT_RE = re.compile(
    r"\\(?:begin|end)\s*\{[^{}]+\}(?:\s*\[[^\]]*\])?",
    re.IGNORECASE,
)
_RESIDUAL_STRUCTURE_COMMAND_RE = re.compile(
    r"\\(?:tableofcontents|listoffigures|listoftables)\b", re.IGNORECASE
)
_ORPHAN_CONNECTORS = {"and", "or", "with", "along with", "where", "whereas", "here"}


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return bool(backslashes % 2)


def strip_comments(text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""

    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "%" and not _is_escaped(text, index):
            newline = text.find("\n", index)
            if newline < 0:
                break
            output.append("\n")
            index = newline + 1
            continue
        output.append(text[index])
        index += 1
    return "".join(output)


def _read_balanced(text: str, start: int, opener: str, closer: str) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != opener:
        return None
    depth = 1
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == opener and not _is_escaped(text, index):
            depth += 1
        elif char == closer and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
        index += 1
    return None


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _parse_heading(text: str, start: int) -> tuple[str, str, int] | None:
    match = _HEADING_COMMAND_RE.match(text, start)
    if not match:
        return None
    index = _skip_space(text, match.end())
    if index < len(text) and text[index] == "[":
        optional = _read_balanced(text, index, "[", "]")
        if optional is None:
            return None
        _, index = optional
        index = _skip_space(text, index)
    title = _read_balanced(text, index, "{", "}")
    if title is None:
        return None
    content, end = title
    return match.group("name").lower(), content.strip(), end


def _is_reference_title(title: str) -> bool:
    plain = re.sub(r"\\[A-Za-z@]+\*?(?:\s*\{([^{}]*)\})?", r"\1", title)
    plain = re.sub(r"[^a-z]+", " ", plain.lower()).strip()
    return bool(re.match(r"^(references|bibliography|literature cited|works cited)(?:\s|$)", plain))


def _find_environment_end(text: str, begin_start: int, environment: str) -> tuple[int, int] | None:
    begin_match = _SPECIAL_BEGIN_RE.match(text, begin_start)
    if not begin_match:
        return None
    depth = 1
    for match in _BEGIN_END_RE.finditer(text, begin_match.end()):
        if match.group("env").lower() != environment.lower():
            continue
        if match.group("action").lower() == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return match.start(), match.end()
    return None


def _extract_document_body(text: str) -> str:
    begin = re.search(r"\\begin\s*\{document\}", text, re.IGNORECASE)
    if not begin:
        return text
    end_matches = list(re.finditer(r"\\end\s*\{document\}", text, re.IGNORECASE))
    end = end_matches[-1].start() if end_matches else len(text)
    return text[begin.end() : end]


def _strip_front_matter_commands(text: str) -> str:
    """Remove balanced title/author/affiliation metadata from document prose."""

    output: list[str] = []
    position = 0
    for match in _FRONT_MATTER_RE.finditer(text):
        if match.start() < position:
            continue
        output.append(text[position : match.start()])
        index = _skip_space(text, match.end())
        # Classes commonly add one or more optional attributes (short author,
        # ORCID, affiliation number).  They belong to the removed command too.
        while index < len(text) and text[index] == "[":
            optional = _read_balanced(text, index, "[", "]")
            if optional is None:
                break
            _, index = optional
            index = _skip_space(text, index)

        valid = True
        for _ in range(_FRONT_MATTER_ARGUMENTS[match.group("name").lower()]):
            argument = _read_balanced(text, index, "{", "}")
            if argument is None:
                valid = False
                break
            _, index = argument
            index = _skip_space(text, index)
        # Elsevier-style author commands may put ORCID metadata after the
        # mandatory name: ``\\author[1]{Name}[orcid=...]``.
        while valid and index < len(text) and text[index] == "[":
            optional = _read_balanced(text, index, "[", "]")
            if optional is None:
                break
            _, index = optional
            index = _skip_space(text, index)
        if valid:
            position = index
        else:
            # Preserve malformed input rather than accidentally discarding prose.
            output.append(match.group(0))
            position = match.end()
    output.append(text[position:])
    return "".join(output)


def _active_heading_levels(text: str) -> dict[str, int]:
    used = {match.group("name").lower() for match in _HEADING_COMMAND_RE.finditer(text)}
    return {name: level for level, name in enumerate(name for name in HEADING_ORDER if name in used)}


def _add_sentences(parent: Node, text: str, split_sentences: SentenceSplitter) -> None:
    if not text or not text.strip():
        return
    # Unknown theorem/proof/center wrappers are layout, not sentence content.
    # Recognized math/list/figure/abstract environments are intercepted before
    # reaching this function, so removing residual delimiters is safe.
    text = _RESIDUAL_ENVIRONMENT_RE.sub(" ", text)
    text = _RESIDUAL_STRUCTURE_COMMAND_RE.sub(" ", text)
    for sentence in split_sentences(text):
        sentence = sentence.strip()
        if sentence and not re.fullmatch(r"(?:\\[A-Za-z@]+\*?\s*)+", sentence):
            parent.add(Node("sentence", sentence))


def _split_items(content: str) -> list[str]:
    """Split list items while ignoring ``\\item`` inside nested environments."""

    starts: list[tuple[int, int]] = []
    depth = 0
    cursor = 0
    token_re = re.compile(r"\\(?:begin|end)\s*\{[^{}]+\}|\\item(?:\s*\[[^\]]*\])?", re.IGNORECASE)
    for match in token_re.finditer(content):
        token = match.group(0).lower()
        if token.startswith("\\begin"):
            depth += 1
        elif token.startswith("\\end"):
            depth = max(0, depth - 1)
        elif depth == 0:
            starts.append((match.start(), match.end()))
        cursor = match.end()
    del cursor
    if not starts:
        return [content] if content.strip() else []
    items: list[str] = []
    for index, (_, body_start) in enumerate(starts):
        body_end = starts[index + 1][0] if index + 1 < len(starts) else len(content)
        item = content[body_start:body_end].strip()
        if item:
            items.append(item)
    return items


def _parse_fragment(parent: Node, text: str, split_sentences: SentenceSplitter) -> None:
    """Parse prose and block environments without changing section context."""

    position = 0
    while position < len(text):
        match = _EVENT_RE.search(text, position)
        if not match:
            _add_sentences(parent, text[position:], split_sentences)
            return
        _add_sentences(parent, text[position : match.start()], split_sentences)
        token = match.group(0)
        lowered = token.lower()

        if lowered.startswith("\\begin"):
            begin = _SPECIAL_BEGIN_RE.match(text, match.start())
            if begin is None:
                position = match.end()
                continue
            environment = begin.group("env").lower()
            end = _find_environment_end(text, match.start(), environment)
            if end is None:
                _add_sentences(parent, token, split_sentences)
                position = match.end()
                continue
            close_start, close_end = end
            inner = text[begin.end() : close_start]
            full = text[match.start() : close_end].strip()
            if environment == "thebibliography":
                return
            if environment in KEYWORD_ENVIRONMENTS:
                pass
            elif environment in ABSTRACT_ENVIRONMENTS:
                abstract = parent.add(Node("abstract", "Abstract"))
                _parse_fragment(abstract, inner, split_sentences)
            elif environment in MATH_ENVIRONMENTS:
                parent.add(Node("formula", full))
            elif environment in FIGURE_ENVIRONMENTS:
                parent.add(Node("figure", full))
            elif environment in LIST_ENVIRONMENTS:
                list_node = parent.add(Node("list", environment))
                for item_text in _split_items(inner):
                    item_node = list_node.add(Node("item", ""))
                    _parse_fragment(item_node, item_text, split_sentences)
            position = close_end
            continue

        if token == "$$":
            close = text.find("$$", match.end())
            if close < 0:
                _add_sentences(parent, token, split_sentences)
                position = match.end()
            else:
                parent.add(Node("formula", text[match.start() : close + 2].strip()))
                position = close + 2
            continue

        if token == "\\[":
            close_match = re.search(r"\\\]", text[match.end() :])
            if close_match is None:
                _add_sentences(parent, token, split_sentences)
                position = match.end()
            else:
                close_end = match.end() + close_match.end()
                parent.add(Node("formula", text[match.start() : close_end].strip()))
                position = close_end
            continue

        # Headings inside list items are unusual; preserve their title as prose.
        heading = _parse_heading(text, match.start())
        if heading:
            _, title, end = heading
            _add_sentences(parent, title, split_sentences)
            position = end
        elif lowered.startswith(("\\bibliography", "\\printbibliography")):
            return
        else:
            position = match.end()


def _prune_orphan_connectors(node: Node) -> None:
    """Drop connector-only fragments split off between block formulas.

    Short sentences are not removed generally: ``Thus:`` and ``Define`` can be
    meaningful formula introductions.  Only a small closed set of connectors is
    dropped when directly adjacent to a formula block.
    """

    for child in node.children:
        _prune_orphan_connectors(child)
    kept: list[Node] = []
    for index, child in enumerate(node.children):
        plain = re.sub(r"[^a-z]+", " ", child.content.lower()).strip()
        adjacent_formula = (
            (index > 0 and node.children[index - 1].kind == "formula")
            or (index + 1 < len(node.children) and node.children[index + 1].kind == "formula")
        )
        if child.kind == "sentence" and plain in _ORPHAN_CONNECTORS and adjacent_formula:
            continue
        kept.append(child)
    node.children = kept


def build_tree(text: str, split_sentences: SentenceSplitter) -> Node:
    """Parse flattened LaTeX into a document tree.

    The hierarchy level is inferred from the heading commands actually present.
    For example, if a paper uses ``section`` and ``subsection`` but no ``chapter``,
    ``section`` becomes the first level below the document root.
    """

    if not callable(split_sentences):
        raise TypeError("split_sentences must be callable")

    body = _strip_front_matter_commands(_extract_document_body(strip_comments(text)))
    levels = _active_heading_levels(body)
    root = Node("document", "Document")
    stack: list[tuple[int, Node]] = [(-1, root)]
    current = root
    position = 0

    while position < len(body):
        match = _EVENT_RE.search(body, position)
        if not match:
            _parse_fragment(current, body[position:], split_sentences)
            break

        _parse_fragment(current, body[position : match.start()], split_sentences)
        token = match.group(0)
        lowered = token.lower()

        heading = _parse_heading(body, match.start())
        if heading:
            name, title, end = heading
            if _is_reference_title(title):
                break
            level = levels.get(name, len(levels))
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else root
            current = parent.add(Node(name, title))
            stack.append((level, current))
            position = end
            continue

        if lowered.startswith(("\\bibliography", "\\printbibliography")):
            break
        if lowered == "\\appendix":
            # The following starred or numbered headings remain included. The
            # marker itself has no semantic text to store.
            position = match.end()
            continue

        if lowered.startswith("\\begin"):
            begin = _SPECIAL_BEGIN_RE.match(body, match.start())
            if begin and begin.group("env").lower() == "thebibliography":
                break

        if lowered.startswith("\\abstract") and not lowered.startswith("\\begin"):
            index = _skip_space(body, match.end())
            argument = _read_balanced(body, index, "{", "}")
            if argument is not None:
                content, position = argument
                abstract = root.add(Node("abstract", "Abstract"))
                _parse_fragment(abstract, content, split_sentences)
                continue

        # Delegate the complete block to the fragment parser, then advance past it.
        if lowered.startswith("\\begin"):
            begin = _SPECIAL_BEGIN_RE.match(body, match.start())
            if begin:
                environment = begin.group("env").lower()
                end = _find_environment_end(body, match.start(), environment)
                if end:
                    close_end = end[1]
                    target = root if environment in ABSTRACT_ENVIRONMENTS else current
                    _parse_fragment(target, body[match.start() : close_end], split_sentences)
                    position = close_end
                    continue
        elif token == "$$":
            close = body.find("$$", match.end())
            if close >= 0:
                _parse_fragment(current, body[match.start() : close + 2], split_sentences)
                position = close + 2
                continue
        elif token == "\\[":
            close_match = re.search(r"\\\]", body[match.end() :])
            if close_match:
                close_end = match.end() + close_match.end()
                _parse_fragment(current, body[match.start() : close_end], split_sentences)
                position = close_end
                continue

        position = match.end()

    _prune_orphan_connectors(root)
    return root

