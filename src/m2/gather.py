"""Find the compilation root of one TeX version and inline its included files.

``flatten(version_dir)`` returns the text TeX would actually read: comments,
``comment`` environments and ``\\iffalse`` blocks are removed, and every
``\\input``/``\\include``/``\\subfile``/``\\import`` reachable from the root is
replaced by the content of the target file. Files outside this compilation path
are reported as unused so the report can count redundant sources.

References used while writing this module:

* TeX comment and end-of-line rules: D. E. Knuth, *The TeXbook*, ch. 8.
* ``\\include``/``\\includeonly``: L. Lamport, *LaTeX: A Document Preparation
  System*, 2nd ed., sec. 4.4.
* ``import`` package (``\\import``/``\\subimport``):
  https://ctan.org/pkg/import
* ``subfiles`` package: https://ctan.org/pkg/subfiles
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ENCODINGS = ("utf-8", "cp1252", "latin-1")
TEX_SUFFIXES = {".tex", ".ltx"}
MAX_INCLUDE_DEPTH = 32
MAX_TEXT_CHARS = 40_000_000

# Environments whose body TeX reads verbatim, so ``%`` is not a comment there.
VERBATIM_ENVIRONMENTS = ("verbatim", "verbatim*", "Verbatim", "lstlisting", "minted", "alltt")

_VERBATIM_COMMAND_RE = re.compile(r"\\(verb\*?|url|path)(?![A-Za-z@])")
_ROOT_RE = re.compile(r"\\document(?:class|style)(?![A-Za-z@])")
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{document\}")
_END_DOCUMENT_RE = re.compile(r"\\end\s*\{document\}")
_SUBFILES_CLASS_RE = re.compile(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{(?:subfiles|standalone)\}")
_VERBATIM_BEGIN_RE = re.compile(
    r"\\begin\s*\{(" + "|".join(re.escape(name) for name in VERBATIM_ENVIRONMENTS) + r")\}"
)
_COMMENT_ENV_RE = re.compile(r"\\begin\s*\{comment\}.*?\\end\s*\{comment\}", re.DOTALL)
_ENDINPUT_RE = re.compile(r"\\endinput(?![A-Za-z@])")
_CONDITIONAL_RE = re.compile(r"\\(?P<name>if[A-Za-z@]*|else|fi)(?![A-Za-z@])")
# Commands that start with ``\if`` but are not TeX conditionals closed by ``\fi``.
_NOT_CONDITIONALS = {"iff", "ifthenelse", "ifboolexpr", "ifstrequal", "ifblank", "ifdefined@"}

_INCLUDE_RE = re.compile(
    r"\\(?P<cmd>input|include|subfile)(?![A-Za-z@])\s*\{(?P<name>[^{}]+)\}"
    r"|\\(?P<icmd>import|subimport|inputfrom|subinputfrom|includefrom|subincludefrom)\*?"
    r"\s*\{(?P<dir>[^{}]*)\}\s*\{(?P<iname>[^{}]+)\}"
    # Plain TeX form: ``\input file`` without braces.
    r"|\\input(?![A-Za-z@{])[ \t]+(?P<bare>[^\s{}\\%]+)"
)
_INCLUDEONLY_RE = re.compile(r"\\includeonly\s*\{(?P<names>[^{}]*)\}")

_MAIN_NAME_RE = re.compile(r"^(main|ms|paper|manuscript|arxiv|article|root)|main", re.IGNORECASE)
_SECONDARY_NAME_RE = re.compile(
    r"supp|(^|[_\-\s])si($|[_\-\s])|appendi|response|rebuttal|reply|letter|cover|checklist"
    r"|summary|poster|slide|abstract|highlight|graphical|standalone|tikz|figure|table"
    r"|review|diff|old|backup|copy|template|sample|example|instruction|readme",
    re.IGNORECASE,
)


@dataclass(slots=True)
class GatherResult:
    """Flattened compilation path of one version.

    Paths are POSIX strings relative to the version directory so the per-version
    statistics written by ``run_m2.py`` are portable between machines.
    """

    root: str | None
    used_files: list[str]
    unused_files: list[str]
    text: str
    missing_files: list[str] = field(default_factory=list)
    cyclic_includes: list[str] = field(default_factory=list)
    non_utf8_files: list[str] = field(default_factory=list)
    root_candidates: list[str] = field(default_factory=list)


def read_tex(path: Path) -> tuple[str, str]:
    """Decode a TeX file, falling back from UTF-8 to legacy 8-bit encodings."""

    data = path.read_bytes()
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.removeprefix("\ufeff"), encoding
    # latin-1 decodes every byte, so this line is only reached on an empty list.
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return bool(backslashes % 2)


def _comment_start(line: str) -> int:
    """Return the index of the first real ``%`` in ``line`` or -1.

    ``\\%`` is an escaped percent sign. ``\\verb|..|`` and ``\\url{..}`` read
    their argument verbatim, so a ``%`` inside them is not a comment either.
    """

    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if char == "\\":
            match = _VERBATIM_COMMAND_RE.match(line, index)
            if match:
                start = match.end()
                if match.group(1).startswith("verb") and start < length:
                    close = line.find(line[start], start + 1)
                    index = length if close < 0 else close + 1
                    continue
                if start < length and line[start] == "{":
                    close = line.find("}", start + 1)
                    index = length if close < 0 else close + 1
                    continue
            index += 2
            continue
        if char == "%":
            return index
        index += 1
    return -1


def strip_comments(text: str) -> str:
    """Remove TeX comments with TeX's end-of-line semantics.

    A comment also swallows its end of line and the leading blanks of the next
    line, as TeX does, so ``word%\\n  next`` becomes ``wordnext``. A blank line
    after a comment still ends the paragraph. Verbatim environments are kept.
    """

    output: list[str] = []
    joining = False
    verbatim_end: re.Pattern[str] | None = None
    for line in text.split("\n"):
        if verbatim_end is not None:
            output.append(line + "\n")
            if verbatim_end.search(line):
                verbatim_end = None
            continue
        if joining:
            if line.strip():
                line = line.lstrip(" \t")
            else:
                output.append("\n")
            joining = False

        begin = _VERBATIM_BEGIN_RE.search(line)
        cut = _comment_start(line) if "%" in line else -1
        if begin and (cut < 0 or begin.start() < cut):
            environment = re.escape(begin.group(1))
            end_re = re.compile(r"\\end\s*\{" + environment + r"\}")
            if not end_re.search(line, begin.end()):
                verbatim_end = end_re
            output.append(line + "\n")
            continue
        if cut >= 0:
            output.append(line[:cut])
            joining = True
        else:
            output.append(line + "\n")
    # split() yields one more piece than there are newlines, and every kept
    # piece was given a newline above; drop the one added after the last piece.
    if output and output[-1].endswith("\n"):
        output[-1] = output[-1][:-1]
    return "".join(output)


def remove_iffalse(text: str) -> str:
    """Drop ``\\iffalse ... \\fi`` blocks, keeping an ``\\else`` branch if present."""

    if "\\iffalse" not in text and "\\if0" not in text:
        return text
    output: list[str] = []
    position = 0
    pattern = re.compile(r"\\iffalse(?![A-Za-z@])|\\if\s*0(?![0-9])")
    while True:
        match = pattern.search(text, position)
        if match is None:
            output.append(text[position:])
            return "".join(output)
        output.append(text[position : match.start()])
        depth = 1
        cursor = match.end()
        else_at: int | None = None
        for token in _CONDITIONAL_RE.finditer(text, cursor):
            name = token.group("name")
            if name == "fi":
                depth -= 1
                if depth == 0:
                    if else_at is not None:
                        output.append(text[else_at : token.start()])
                    position = token.end()
                    break
            elif name == "else":
                if depth == 1 and else_at is None:
                    else_at = token.end()
            elif name not in _NOT_CONDITIONALS:
                depth += 1
        else:
            # Unterminated \iffalse: TeX would skip to the end of the file.
            return "".join(output)


def preprocess(text: str) -> str:
    """Remove text TeX never typesets before includes are followed."""

    text = strip_comments(text)
    text = _COMMENT_ENV_RE.sub("", text)
    text = remove_iffalse(text)
    end = _ENDINPUT_RE.search(text)
    return text[: end.start()] if end else text


def _document_body(text: str) -> str:
    begin = _BEGIN_DOCUMENT_RE.search(text)
    if not begin:
        return text
    ends = list(_END_DOCUMENT_RE.finditer(text, begin.end()))
    return text[begin.end() : ends[-1].start() if ends else len(text)]


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


class _Flattener:
    """Expand includes for one root file inside one version directory."""

    def __init__(
        self, version_dir: Path, tex_files: list[Path], cache: dict[Path, tuple[str, str]]
    ) -> None:
        self.version_dir = version_dir
        self.tex_files = tex_files
        self.cache = cache
        self.used: list[Path] = []
        self.missing: list[str] = []
        self.cyclic: list[str] = []
        self.include_only: set[str] | None = None
        self.size = 0
        self._by_lower = {_relative(path, version_dir).lower(): path for path in tex_files}
        self._by_name: dict[str, list[Path]] = {}
        for path in tex_files:
            self._by_name.setdefault(path.name.lower(), []).append(path)

    def load(self, path: Path) -> str:
        if path not in self.cache:
            raw, encoding = read_tex(path)
            self.cache[path] = (preprocess(raw), encoding)
        return self.cache[path][0]

    def _inside(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.version_dir.resolve())
        except (OSError, ValueError):
            return False
        return True

    def resolve(self, name: str, bases: list[Path]) -> Path | None:
        name = name.strip().strip('"').strip()
        if not name:
            return None
        candidates = [name]
        if Path(name).suffix.lower() not in TEX_SUFFIXES:
            candidates.insert(0, name + ".tex")
        for base in bases:
            for candidate in candidates:
                path = base / candidate
                if path.is_file() and self._inside(path):
                    return path
        # Fall back to case-insensitive and, if unique, basename matching. Both
        # cover archives packed on case-insensitive or flattened file systems.
        for base in bases:
            for candidate in candidates:
                key = _relative((base / candidate).resolve(), self.version_dir.resolve()).lower()
                if key in self._by_lower:
                    return self._by_lower[key]
        for candidate in candidates:
            matches = self._by_name.get(Path(candidate).name.lower(), [])
            if len(matches) == 1:
                return matches[0]
        return None

    def expand(self, path: Path, stack: tuple[Path, ...], import_dir: Path, is_root: bool) -> str:
        if path not in self.used:
            self.used.append(path)
        text = self.load(path)
        if not is_root and _BEGIN_DOCUMENT_RE.search(text):
            # Subfiles and standalone children: only the body is typeset.
            text = _document_body(text)
        if is_root:
            include_only = _INCLUDEONLY_RE.search(text)
            if include_only:
                names = [name.strip() for name in include_only.group("names").split(",")]
                self.include_only = {name.removesuffix(".tex") for name in names if name}
        root_dir = stack[0].parent if stack else path.parent

        def replace(match: re.Match[str]) -> str:
            command = match.group("cmd") or match.group("icmd") or "input"
            if match.group("icmd"):
                directory = match.group("dir").strip()
                name = match.group("iname")
                if command.startswith("sub"):
                    bases = [path.parent / directory]
                else:
                    bases = [import_dir / directory, root_dir / directory]
            else:
                name = match.group("name") or match.group("bare")
                # LaTeX resolves \input against the root directory; some authors
                # assume the including file's directory instead, so try both.
                bases = [import_dir, root_dir, path.parent]
            if command == "include" and self.include_only is not None:
                if name.strip().removesuffix(".tex") not in self.include_only:
                    return ""
            target = self.resolve(name, bases)
            if target is None:
                suffix = Path(name.strip()).suffix.lower()
                if suffix in TEX_SUFFIXES or not suffix:
                    self.missing.append(name.strip())
                return ""
            if target in stack:
                self.cyclic.append(_relative(target, self.version_dir))
                return ""
            if len(stack) >= MAX_INCLUDE_DEPTH or self.size > MAX_TEXT_CHARS:
                return ""
            child_import_dir = target.parent if match.group("icmd") or command == "subfile" else import_dir
            content = self.expand(target, stack + (target,), child_import_dir, False)
            self.size += len(content)
            # \include starts a new page, i.e. it always ends the paragraph.
            return f"\n{content}\n" if command == "include" else content

        return _INCLUDE_RE.sub(replace, text)


def _list_tex_files(version_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in version_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TEX_SUFFIXES
    )


def _root_rank(path: Path, text: str, reachable_chars: int, version_dir: Path) -> tuple:
    stem = path.stem
    return (
        bool(_BEGIN_DOCUMENT_RE.search(text)),
        not _SUBFILES_CLASS_RE.search(text),
        not _SECONDARY_NAME_RE.search(stem),
        bool(_MAIN_NAME_RE.search(stem)),
        path.parent == version_dir,
        reachable_chars,
    )


def find_root(version_dir: str | Path) -> Path | None:
    """Return the compilation root of a version directory, or ``None``."""

    version_dir = Path(version_dir)
    tex_files = _list_tex_files(version_dir)
    return _choose_root(version_dir, tex_files, {})[0]


def _choose_root(
    version_dir: Path, tex_files: list[Path], cache: dict[Path, tuple[str, str]]
) -> tuple[Path | None, list[Path]]:
    if not tex_files:
        return None, []
    probe = _Flattener(version_dir, tex_files, cache)
    candidates = [path for path in tex_files if _ROOT_RE.search(probe.load(path))]
    if not candidates:
        candidates = [path for path in tex_files if _BEGIN_DOCUMENT_RE.search(probe.load(path))]
    if not candidates:
        # No LaTeX root at all (rare plain-TeX uploads): use the largest file.
        return max(tex_files, key=lambda path: len(probe.load(path))), []
    if len(candidates) == 1:
        return candidates[0], candidates

    reachable: dict[Path, set[Path]] = {}
    ranks: dict[Path, tuple] = {}
    for candidate in candidates:
        flattener = _Flattener(version_dir, tex_files, cache)
        text = flattener.expand(candidate, (candidate,), candidate.parent, True)
        reachable[candidate] = set(flattener.used) - {candidate}
        ranks[candidate] = _root_rank(candidate, probe.load(candidate), len(text), version_dir)
    # A candidate included by another candidate is a child document, not a root.
    independent = [
        candidate
        for candidate in candidates
        if not any(candidate in reachable[other] for other in candidates if other != candidate)
    ] or candidates
    return max(independent, key=lambda path: ranks[path]), candidates


def flatten(version_dir: str | Path) -> GatherResult:
    """Gather the compilation path of ``version_dir`` into one string."""

    version_dir = Path(version_dir)
    tex_files = _list_tex_files(version_dir) if version_dir.is_dir() else []
    cache: dict[Path, tuple[str, str]] = {}
    root, candidates = _choose_root(version_dir, tex_files, cache)
    if root is None:
        return GatherResult(None, [], [], "")

    flattener = _Flattener(version_dir, tex_files, cache)
    text = flattener.expand(root, (root,), root.parent, True)
    used = set(flattener.used)
    return GatherResult(
        root=_relative(root, version_dir),
        used_files=[_relative(path, version_dir) for path in flattener.used],
        unused_files=[_relative(path, version_dir) for path in tex_files if path not in used],
        text=text,
        missing_files=sorted(set(flattener.missing)),
        cyclic_includes=sorted(set(flattener.cyclic)),
        non_utf8_files=[
            _relative(path, version_dir) for path in flattener.used if cache[path][1] != "utf-8"
        ],
        root_candidates=[_relative(path, version_dir) for path in candidates],
    )
