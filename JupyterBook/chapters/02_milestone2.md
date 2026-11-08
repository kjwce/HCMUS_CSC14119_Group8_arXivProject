---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Milestone 2: Parsing & Standardization

## 1. Hierarchical Parsing and Standardization Methodology

Raw LaTeX sources are written for a typesetting engine rather than for data analysis. A single paper revision may be split across several included files, contain commented-out draft text, mix different math delimiters, format references manually with `\bibitem`, or rename citation keys between revisions.

To turn the Milestone 1 corpus into the `elements`, `hierarchy`, and `bibtex` tables, our parser (`src/m2/`) processes all versions ($v_1, \dots, v_N$) of each publication in two passes.

```{mermaid}
flowchart TD
    subgraph Pass1["Pass 1: Per-Version Gathering & Bibliography Extraction"]
        A["Milestone 1 Paper Directory<br/>metadata.json + v1..vN/"] --> B{"Usable TeX source<br/>present?"}
        B -->|No| C["Record Empty Version<br/>(pdf_only, no_source, etc.)"]
        B -->|Yes| D["Compilation-Path Flattening<br/>(Root detection + transitive includes)"]
        D --> E["Bibliography Extraction<br/>(.bib files + \\bibitem blocks)"]
    end

    subgraph Sync["Cross-Version Synchronization Barrier"]
        E --> F["Reference Deduplication & Field Union<br/>(Constrained Union-Find across v1..vN)"]
        F --> G["Unified BibTeX Records"]
        F --> H["Per-Version Citation Rename Maps<br/>(old_key → chosen_key)"]
    end

    subgraph Pass2["Pass 2: Tree Construction, Normalization & DAG Storage"]
        D --> I["Hierarchy Tree Construction<br/>(Monotonic heading stack + sentence splitter)"]
        I --> J["In-Place Node Standardization<br/>(rename_cites → normalize_math → clean)"]
        H --> J
        J --> K["Content-Addressed Deduplication<br/>(SHA-256 element store + version edges)"]
        C --> K
        K --> L["Atomic Per-Paper Output<br/>elements, hierarchy, bibtex, stats"]
    end
```

Running two passes across versions solves an ordering dependency between reference deduplication and full-text deduplication:
1. Pass 1 (Gathering and Bibliography Extraction): For each available version, the parser finds the compilation root, inlines all transitively included TeX files, and extracts all bibliography entries.
2. Cross-Version Bibliography Deduplication: Before building any version's document tree, extracted bibliography entries across $v_1, \dots, v_N$ are merged. Each referenced work receives a single canonical citation key, producing per-version rename maps.
3. Pass 2 (Tree Construction, Normalization, and Storage): Each version's flattened LaTeX is parsed into a hierarchy tree, standardized in place (citation renaming $\to$ math normalization $\to$ formatting cleanup), and added to the shared element store. Because citation keys are unified before hashing, a sentence whose only change between $v_1$ and $v_2$ was a renamed citation key normalizes to the same string and shares one element ID.

| Module | Role in the Milestone 2 Pipeline |
|---|---|
| `models.py` | Defines the `Node` dataclass (kind, text content, and child list) shared across parsing and normalization. |
| `gather.py` | Decodes TeX files with encoding fallback, strips comments and dead conditionals, detects the root TeX file, and inlines included files. |
| `hierarchy.py` | Strips front-matter metadata, infers active heading depths, builds the hierarchy tree via a monotonic stack, excludes reference sections, and segments leaves. |
| `normalize.py` | Implements protected-span sentence splitting, citation key renaming, math delimiter unification, and formatting cleanup. |
| `bib.py` | Extracts BibTeX and `\bibitem` records (using a fast scanner with a `bibtexparser` fallback) and deduplicates references across versions. |
| `store.py` | Assigns SHA-256 element IDs with intra-version occurrence ordinals and atomically writes the per-publication tables and statistics. |
| `run_m2.py` | Manages multi-process execution, schema-versioned checkpointing, corpus inventory checks, and memory sampling. |

---

### 1.1. Input Corpus Preprocessing and Empty-Version Handling

#### Corpus Discovery and Empty-Version Convention
The parser scans the Milestone 1 raw directory for paper folders containing `metadata.json` and normalizes each arXiv ID to the hyphenated form (e.g., 2605.27338 becomes 2605-27338). For each paper, it reads the ordered version list ($v_1, v_2, \dots$) and the crawling status recorded in Milestone 1.

About 5% of arXiv revisions have no usable LaTeX source (PDF-only uploads, HTTP 404 withdrawals, non-TeX files, or download errors) and sit in empty version folders. Following the project specification, the parser neither skips these versions nor invents fake document roots for them. Instead, every version without compilable TeX is given a diagnostic status and recorded as an empty version in the store:

```python
for spec in specs:
    version = _version_number(spec["version"])
    status = str(spec.get("source_status") or "unknown")
    version_dir = paper_dir / f"v{version}"
    if status != "ok" or not version_dir.is_dir():
        version_status[version] = "empty" if status != "ok" else "missing_directory"
        continue
    result = gather_module.flatten(version_dir)
    if not getattr(result, "text", "").strip():
        version_status[version] = "empty_gathered_text"
        continue
```

Recording an empty version increments the seen version count and logs the version number in the empty-version list, while leaving the parsed version count and output tables untouched:

| Version Condition | Recorded Status | Versions Seen | Versions Parsed | Rows Added to Tables |
|---|---|:---:|:---:|:---:|
| Non-TeX or missing source in Milestone 1 | *empty* | $+1$ | $+0$ | $0$ |
| Marked available in metadata, but folder missing | *missing_directory* | $+1$ | $+0$ | $0$ |
| Folder exists, but flattened text is empty | *empty_gathered_text* | $+1$ | $+0$ | $0$ |
| Flattened and parsed into a document tree | *parsed* | $+1$ | $+1$ | $\ge 1$ |

#### Character Encoding Fallback
While most TeX files use UTF-8, older or international manuscripts occasionally use 8-bit encodings such as Windows-1252 or Latin-1 for accented characters (e.g., *Café* or *Müller*). Our file reader decodes files in priority order—UTF-8 first (stripping any byte-order mark), then Windows-1252, and finally Latin-1—logging any non-UTF-8 files in the gather statistics.

#### Pre-Include Dead-Code Stripping
Before searching a file for `\documentclass` or following inclusion commands, `preprocess()` strips regions that TeX itself ignores:

```python
def preprocess(text: str) -> str:
    text = strip_comments(text)
    text = _COMMENT_ENV_RE.sub("", text)
    text = remove_iffalse(text)
    end = _ENDINPUT_RE.search(text)
    return text[: end.start()] if end else text
```

1. Comment stripping: Following TeX's end-of-line rules (Knuth, *The TeXbook*, Ch. 8), an unescaped `%` removes the rest of the line, the newline character, and leading spaces or tabs on the next line—unless the next line is blank, in which case the paragraph break is kept. Literal `%` characters inside verbatim environments (*verbatim*, *lstlisting*, *minted*, *alltt*), inline verbatim commands (`\verb`, `\url`, `\path`), and escaped `\%` tokens are preserved, while even backslash sequences (`\\%`) are recognized as a line break `\\` followed by a comment.
2. Comment environments: Removes `\begin{comment}...\end{comment}` blocks.
3. Conditional dead branches: Authors often disable blocks of text or old file includes with `\iffalse ... \fi` or `\if0 ... \fi`. Our scanner tracks nesting depth across `\if*`, `\else`, and `\fi` tokens, dropping the false branch, keeping a top-level `\else` branch if present, and ignoring non-conditional macros that start with the letters *if* (such as `\iff` or `\ifthenelse`).
4. Early file termination: Truncates everything after `\endinput`.

| Preprocessing Rule | Raw TeX Input | Preprocessed Output | Effect |
|---|---|---|---|
| Comment + line join | `50\% of runs % comment\n  succeeded` | `50\% of runs succeeded` | Keeps `\%` and joins across the commented newline. |
| Even backslash + `%` | `a\\% hidden comment\nb` | `a\\b` | Treats `\\` as a line break and `%` as a comment. |
| Comment + blank line | `para one%\n\npara two` | `para one\n\npara two` | Preserves paragraph breaks for sentence splitting. |
| Verbatim protection | `\url{http://a.org/x%20y} and \verb|50%|` | `\url{http://a.org/x%20y} and \verb|50%|` | Does not truncate lines at `%` inside URLs or verbatim text. |
| Nested `\iffalse` + `\else` | `keep \iffalse drop \ifx\a\b x\fi \else live\fi` | `keep live` | Tracks nested conditionals and keeps the `\else` branch. |

---

### 1.2. Publication Root `.tex` Detection

A multi-file arXiv submission often contains several TeX files with `\documentclass` declarations: the main paper, a supplementary file (*SI.tex*), a reviewer response, standalone TikZ figures, or *subfiles* chapters. Choosing the largest file fails when a supplement full of tables is longer than the main paper.

To select the true compilation root, `_choose_root()` uses a three-stage algorithm:
1. Candidate discovery: Finds all preprocessed TeX files containing `\documentclass` or `\documentstyle`. If none exist, it falls back to files with `\begin{document}`, and finally to the largest TeX file for plain-TeX submissions.
2. Transitive reachability pruning: Trial-expands each candidate and removes any candidate that is transitively included by another candidate (such as a chapter file included via `\subfile`).
3. Six-criteria lexicographic ranking: Scores each remaining independent candidate with a 6-tuple compared from highest to lowest priority:

```python
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
```

| Priority | Criterion | Rule Matched | Purpose |
|:---:|---|---|---|
| 1 | Has document body | Contains `\begin{document}` | Ranks complete manuscripts above preamble-only files. |
| 2 | Not a subfile or standalone figure | Class is neither *subfiles* nor *standalone* | Demotes standalone TikZ figures and unreferenced subfiles. |
| 3 | Not a secondary document | Filename avoids *supp*, *si*, *appendi*, *rebuttal*, *reply*, *letter*, *cover*, *poster*, *slide*, *tikz*, *diff*, *old*, *template*, *readme* | Prevents supplements, rebuttals, or diff files from beating the main paper. |
| 4 | Canonical main filename | Filename matches *main*, *ms*, *paper*, *manuscript*, *arxiv*, *article*, or *root* | Prefers standard main-manuscript names. |
| 5 | Top-level directory location | File sits directly in the version folder | Prefers top-level entry points over subdirectory files. |
| 6 | Expanded character count | Total character length after transitive expansion | Breaks ties in favor of the most complete compiled text. |

For example, if a version folder contains *main.tex* (12,000 expanded chars, includes *chap/one.tex*), *chap/one.tex* (4,000 chars, *subfiles* class), and *SI.tex* (25,000 chars), Stage 2 prunes *chap/one.tex* because it is reached from *main.tex*, and Stage 3 ranks *main.tex* `(True, True, True, True, True, 12000)` above *SI.tex* `(True, True, False, False, True, 25000)` at Priority 3.

---

### 1.3. Multi-File Gathering via Transitive `\input` and `\include` Resolution

Once the root file is chosen, `flatten()` recursively replaces reachable inclusion commands with the target file's preprocessed text. Any TeX file in the version folder not visited on this path is recorded as unused and ignored.

```{mermaid}
flowchart TD
    A["All .tex/.ltx files in version folder"] --> B["Preprocess each file<br/>(strip comments, \\iffalse, \\endinput)"]
    B --> C["Find \\documentclass candidates"]
    C --> D["Prune candidates reachable<br/>from another candidate"]
    D --> E["Select highest-ranked root<br/>(6-criteria lexicographic rank)"]
    E --> F["Expand root recursively"]
    F --> G{"Match inclusion directive<br/>(\\input, \\include, \\subfile, \\import)"}
    G -->|Target resolved & acyclic| H["Strip child preamble if subfile<br/>& inline target recursively"]
    G -->|Missing or cyclic| I["Log diagnostic & replace with empty string"]
    H --> J["Return flattened compilation text<br/>+ used and unused file lists"]
    I --> J
```

#### Supported Inclusion Directives and Path Resolution
The flattener supports five inclusion forms:

| Inclusion Family | Example Syntax | Behavior |
|---|---|---|
| Standard inline | `\input{sections/intro}` | Splices target text inline; searches import, root, and current file directories. |
| Plain-TeX inline | `\input macros` (no braces) | Splices target text inline for plain-TeX macro files. |
| Page-breaking include | `\include{appendix}` | Wraps spliced text in newlines (since `\include` ends a page/paragraph) and honors `\includeonly`. |
| Subfiles package | `\subfile{chap/one}` | Strips the child's preamble up to `\begin{document}` and splices only its body. |
| Import package family | `\import{parts/}{two}` | Joins directory and filename arguments and updates the base directory for nested includes. |

Target paths are resolved through three fallbacks while preventing directory traversal outside the version folder:
1. Exact relative path: Checks candidate base directories, appending *.tex* when no TeX suffix is given. Checking both the root folder and the including file's folder covers authors who write paths relative to child subdirectories.
2. Case-insensitive path fallback: Matches paths across case mismatches (e.g., `\input{Sections/Intro}` for *sections/intro.tex*).
3. Unique basename fallback: Recovers flattened archives when a single file in the version directory matches the target's basename.

To guard against malformed inputs, the expander breaks inclusion cycles using the active call stack, caps recursion at depth 32 and total text at 40 million characters, and logs missing or cyclic targets in the gather statistics.

---

### 1.4. Document Hierarchy Tree Construction

#### The Node Data Contract and Front-Matter Stripping
`build_tree()` parses the flattened text into a tree of `Node(kind, content, children)` objects. Each node's `content` holds only its own label (e.g., a section node stores *"Introduction"*, while its sentences and subsections live in `children`), and no element ID is assigned until after normalization and cleanup.

After slicing the body between `\begin{document}` and `\end{document}`, the parser removes front-matter commands that templates such as *revtex4*, *elsarticle*, *IEEEtran*, and *acmart* place inside the document body before `\maketitle`:

```latex
\begin{document}
\title{A Study of Citation Matching}
\author[1]{Jane Doe}[orcid=0000-0001-2345-6789]
\affiliation{Department of Computer Science}
\cortext[1]{Corresponding author}
\maketitle
\begin{abstract} ... \end{abstract}
```

Using a balanced-brace reader, `_strip_front_matter_commands()` strips 35 metadata commands (including `\title`, `\author`, `\affiliation`, `\address`, `\institute`, `\email`, `\cortext`, `\thanks`, `\date`, `\received`, `\keywords`, and `\maketitle`) along with their preceding optional brackets, mandatory braced arguments, and trailing optional brackets (such as Elsevier ORCID tags). If braces are unbalanced, the token is left untouched so body text is not lost.

#### Adaptive Heading Levels, Monotonic Stack, and Lists
LaTeX defines seven heading ranks from `\part` down to `\subparagraph`. Because articles start at `\section` while books start at `\part` or `\chapter`, `_active_heading_levels()` scans which heading commands actually appear in the document and maps them to contiguous levels $0, 1, 2, \dots$. `build_tree()` then builds the tree in one left-to-right pass with a monotonic ancestor stack initialized to `[(-1, root)]`:

```python
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
```

Two additional container types branch into child nodes:
- Abstracts: Both `\begin{abstract}...\end{abstract}` and `\abstract{...}` attach an *abstract* node under the document root and parse its text into sentence and formula children.
- Itemized lists: Environments (*itemize*, *enumerate*, *description*, *highlights*) create a *list* node. `_split_items()` tracks environment nesting depth and splits only on top-level `\item` tokens (depth 0), creating an *item* node for each point whose body is recursively parsed into sentences, formulas, figures, or nested lists.

```{mermaid}
flowchart TD
    Root["document: 'Document'"] --> Abs["abstract: 'Abstract'"]
    Abs --> S0["sentence: 'We study citation matching.'"]
    Root --> Sec1["section: 'Introduction'"]
    Sec1 --> S1["sentence: 'Prior work uses PDFs.'"]
    Sec1 --> Sub1["subsection: 'Contributions'"]
    Sub1 --> L1["list: 'itemize'"]
    L1 --> I1["item: ''"]
    I1 --> S2["sentence: 'First contribution.'"]
    L1 --> I2["item: ''"]
    I2 --> S3["sentence: 'Second contribution.'"]
    I2 --> F1["formula: '\\begin{equation}...'"]
```

| Node Kind | Tree Role | Stored Content | Permitted Children |
|---|---|---|---|
| *document* | Root (depth 0) | *"Version N root"* (in store) | *abstract*, headings, *list*, leaf nodes |
| *part* $\dots$ *subparagraph* | Section branch | Heading title | Finer headings, *list*, leaf nodes |
| *abstract* | Branch under root | *"Abstract"* | *sentence*, *formula*, *figure*, *list* |
| *list* | List container | Environment name (e.g., *"itemize"*) | *item* |
| *item* | List point | Empty string (content lives in children) | *sentence*, *formula*, *figure*, nested *list* |
| *sentence*, *formula*, *figure* | Leaf nodes | Sentence text, math block, or figure/table block | None |

---

### 1.5. Leaf-Level Segmentation (Sentences, Block Formulas, Figures, and Tables)

Within each container, `_parse_fragment()` segments content into formula, figure, and sentence leaves.

#### Block Formulas and Standalone Figures/Tables
1. Block formulas: Display math delimited by `$$...$$`, `\[...\]`, or any of the 13 standard math environments (*equation*, *align*, *gather*, *multline*, *eqnarray*, *alignat*, *displaymath*, and starred forms) becomes a *formula* leaf.
2. Figures and tables: Tables are stored under the *figure* kind as required. Because authors often use tabular environments without an outer `\begin{table}` float (such as AASTeX *deluxetable\** or a *tabularx* inside *center*), the parser matches 25 environments: outer floats (*figure*, *table*, *wrapfigure*, *wraptable*, *sidewaysfigure*, *sidewaystable*, *subfigure*, *subtable*) and standalone tables (*tabular*, *tabular\**, *tabularx*, *tabulary*, *tblr*, *longtblr*, *talltblr*, *deluxetable*, *deluxetable\**, *supertabular*, *xtabular*, *threeparttable*, *ruledtabular*, *longtable*). Matching an outer *table* float advances the cursor past `\end{table}` so inner *tabular* blocks are not double-counted, while floatless tables are still captured on their own.
3. Orphan connector pruning: When a sentence spans two display equations, extracting the formulas can leave a single connecting word stranded between them. `_prune_orphan_connectors()` drops seven connectors (*and*, *or*, *with*, *along with*, *where*, *whereas*, *here*) only when directly adjacent to a formula sibling, while keeping meaningful intros like *"Thus:"*.

#### Protected-Span Sentence Segmentation
Prose between block environments has residual layout wrappers (e.g., *theorem* or *proof* tags) removed and is split by `split_sentences()` in two phases:

1. Byte-mask protection (`_protected_spans`): Marks character indices where a split is forbidden, covering URLs, inline and display math (`$...$`, `\(...\)`, *math* environments), and LaTeX commands with their balanced `[...]` and `{...}` arguments. Inline text-formatting commands (`\emph`, `\textbf`, `\textit`, `\underline`, etc.) have only their command name masked while leaving their braced text unprotected, so run-in bold headings like `\textbf{Contributions.} We show...` split after *Contributions.*.
2. Boundary verification (`_is_boundary`): For each unmasked `.`, `!`, or `?`:
   - Absorbs closing quotes/brackets and attached trailing commands (`\footnote{...}`, `\cite{...}`) into the ending sentence.
   - Requires following whitespace (protecting decimals like 3.14) and rejects cuts before lowercase letters or `,;:`.
   - Rejects cuts after 45 unconditional abbreviations (*e.g.*, *i.e.*, *et al.*, *Fig.*, *Eq.*, *Sec.*, *Dr.*, month names), after 23 context-sensitive words (*app*, *no*, *ref*, *vol*, *p*, *step*) when followed by a digit or reference opener, and after single uppercase initials (*J. Smith*).
   - Forces a cut at any unmasked blank line (`\n\n`).

| Input LaTeX Prose Snippet | Segmented Sentence Leaves | Rule Illustrated |
|---|---|---|
| `We follow Vaswani et al. \cite{a,b} in Fig. 1. The value is 3.14, i.e. small.` | 1. We follow Vaswani et al. `\cite{a,b}` in Fig. 1.<br/>2. The value is 3.14, i.e. small. | Abbreviations, citations, and decimals (3.14) do not split. |
| `Let $x = 1. Y$ hold. End.\footnote{A note. Two.} Next one.` | 1. Let `$x = 1. Y$` hold.<br/>2. End.`\footnote{A note. Two.}`<br/>3. Next one. | Math and footnote bodies are masked; trailing footnotes stay attached. |
| `We built an app. It works. See No. 5 and Ref. \cite{x}.` | 1. We built an app.<br/>2. It works.<br/>3. See No. 5 and Ref. `\cite{x}`. | *app.* splits before *"It"*, while *No.* (before 5) and *Ref.* (before `\cite`) do not. |
| `\textbf{Contributions.} We show two results.` | 1. `\textbf{Contributions.}`<br/>2. We show two results. | Unmasked text-formatting arguments allow run-in headings to split. |

---

### 1.6. Structural Exclusions (References) and Inclusions (Acknowledgements and Appendices)

The hierarchy builder excludes reference sections (which belong in the `bibtex` table) and keyword metadata while keeping unnumbered Acknowledgements and Appendices.

| Document Division | Specification Rule | Parser Behavior |
|---|---|---|
| References / Bibliography | Exclude | Stops tree construction at reference headings (*References*, *Bibliography*, *Literature Cited*, *Works Cited*), *thebibliography* environments, or `\bibliography` / `\printbibliography` commands. |
| Keywords block | Exclude | Skips *keyword* and *keywords* environments. |
| Acknowledgements | Include | Parses starred headings (`\section*{Acknowledgements}`) at the same level as numbered headings. |
| Appendices | Include | Consumes the `\appendix` switch silently and continues parsing all subsequent appendix sections into the tree. |

Reference headings are detected by stripping inner formatting macros from the heading title and checking whether the normalized lowercase text starts with *references*, *bibliography*, *literature cited*, or *works cited*. Starred headings are supported across all seven sectioning ranks via an optional `*` in the heading pattern. When `\appendix` is encountered, the marker itself is skipped so it does not leak into prose, and all following sections remain attached to the tree.

---

### 1.7. LaTeX Formatting Cleanup and Mathematical Notation Normalization

After a version's tree is built, `_normalize_tree()` standardizes every node's text in the order:

$$\text{rename\_cites} \longrightarrow \text{normalize\_math} \longrightarrow \text{clean}$$

#### Mathematical Notation Normalization
`normalize_math()` unifies inline math (`\(...\)` and `\begin{math}...\end{math}`) to `$...$` and display math (`$$...$$`, `\[...\]`, and 15 environments including *equation\**, *displaymath*, *align*, *alignat*, *flalign*, *gather*, *multline*, and *eqnarray*) to `\begin{equation}...\end{equation}`.

Because LaTeX's *equation* environment forbids top-level alignment tabs (`&`) and line breaks (`\\`), `_normalized_equation()` wraps multi-line aligned bodies inside an inner *aligned* environment and unaligned multi-line bodies (*gather*, *multline*) inside *gathered*:

```python
def _normalized_equation(body: str, wrapper: str | None = None) -> str:
    body = body.strip()
    if wrapper is None and _UNESCAPED_ALIGNMENT_RE.search(body) and not _INNER_MATH_STRUCTURE_RE.search(body):
        wrapper = "aligned"
    if wrapper:
        body = f"\\begin{{{wrapper}}}{body}\\end{{{wrapper}}}"
    return f"\\begin{{equation}}{body}\\end{{equation}}"
```

In addition, the delimiter matcher checks `\\` before `\[` so a line break with vertical spacing like `\\[2pt]` is never misread as opening display math.

#### LaTeX Formatting Cleanup
`clean()` removes layout commands while keeping table and equation delimiters (`&` and `\\`):
1. Float options: Removes placement specifiers like `[htbp]` on floats and width arguments on *wrapfigure* and *wraptable*.
2. Commands removed with arguments (27 commands): Deletes layout, spacing, color, and label commands along with their balanced arguments (`\label`, `\vspace`, `\hspace`, `\setlength`, `\cmidrule`, `\cline`, `\rowcolor`, `\cellcolor`, `\captionsetup`, `\fontsize`).
3. Commands unwrapped to their text argument (18 commands): Replaces inline styling and scaling wrappers (`\textbf`, `\textit`, `\emph`, `\texttt`, `\underline`, `\mbox`, `\textcolor`, `\scalebox`, `\resizebox`) recursively with their final mandatory argument.
4. Formatting switches and spacing (64 commands): Removes argumentless font, alignment, and table-rule switches (`\centering`, `\noindent`, `\small`, `\large`, `\toprule`, `\midrule`, `\bottomrule`, `\hline`, `\nonumber`, `\notag`), converts non-breaking `~` to spaces outside URLs, and collapses whitespace runs to a single space.

| Stage | Before Normalization & Cleanup | After Normalization & Cleanup |
|---|---|---|
| Inline & display math | `Let \(x > 0\) in \[ a = b \]` | `Let $x > 0$ in \begin{equation}a = b\end{equation}` |
| Multi-line align block | `\begin{align*}\label{eq:1} x &= y \nonumber \\ z &= w \end{align*}` | `\begin{equation}\begin{aligned}x &= y \\ z &= w\end{aligned}\end{equation}` |
| Figure float cleanup | `\begin{figure*}[htbp] \centering \small \includegraphics{f.png} \vspace{-2mm} \caption{A \textbf{bold} cap~\ref{x}.\label{fig:x}} \end{figure*}` | `\begin{figure*} \includegraphics{f.png} \caption{A bold cap \ref{x}.} \end{figure*}` |
| Table rule cleanup | `\begin{tabular}{lc}\toprule A & B \\ \midrule \cmidrule(lr){1-2} 1 & 2 \\ \bottomrule\end{tabular}` | `\begin{tabular}{lc} A & B \\ 1 & 2 \\ \end{tabular}` |

---

### 1.8. Programmatic Conversion of `\bibitem` to Standardized BibTeX

During Pass 1, `extract()` converts both external `.bib` databases and embedded `\bibitem` entries into uniform `Entry` records (`citation_key`, `entry_type`, `title`, `authors`, `year`, `doi`, `journal`, `url`, and internal `eprint`).

#### Fast Two-Tier BibTeX Parsing
Standard `bibtexparser` (v1.x) parses only ~40 KB/s (taking 2,125 s on 150 corpus files), and shared `.bib` files often contain thousands of uncited entries. We use a two-tier approach:
1. Cited-key pre-filtering (`select_entries`): Unless `\nocite{*}` is present, slices the `.bib` file to keep only `@string`, `@preamble`, and entries whose keys are actually cited in the text (5–20$\times$ reduction).
2. Custom scanner with fallback (`parse_bib_text`): `fast_bib_parse()` scans `@string` macros, `#` concatenation, and nested braces/quotes in 6 seconds on the same 150-file benchmark (~350$\times$ faster). Any entry that fails the fast scanner is retried with `bibtexparser`.

#### Rule-Based Conversion of `\bibitem` Entries
`_parse_bibitem_body()` converts free-text `\bibitem` entries in three steps:
1. Identifier and masked-year extraction: Extracts DOIs (preserving balanced parentheses in SICI DOIs), new and old arXiv IDs, and URLs. Before searching for a 4-digit year, it masks out URLs, DOIs, arXiv IDs, and `[Accessed: ...]` notes so access dates or arXiv prefixes (e.g., 2010 in *arXiv:2010.12345*) are not mistaken for publication years.
2. Six-style structural cascade: Separates authors, title, and venue by matching six common citation layouts in order:

| Style Family | Trigger Pattern | Field Extraction Logic |
|---|---|---|
| 1. REVTeX macros | `\bibinfo{field}{value}` | Reads *author*, *title*, *journal* / *booktitle*, *year*, *doi*, and *eprint* tags directly. |
| 2. Natbib blocks | `\newblock` dividers | Block 1 $\to$ authors; Block 2 $\to$ title; Block 3 $\to$ venue (italicized text or before comma). |
| 3. IEEE / Quoted | `` ``Title,'' `` or `"Title"` | Before quote $\to$ authors; inside quote $\to$ title; after quote $\to$ venue. |
| 4. APA / Author-Year | `(YYYY)` before italic/bold | Before `(YYYY)` $\to$ authors; after `(YYYY)` split into title and venue at the first top-level period or italic block. |
| 5. Surname-First | *Surname, I., Other, J.* | Scans past surname-initial pairs to split authors from title and italicized venue. |
| 6. Math & Physics | *I. Surname, J. Other, ...* | Splits after final *and* / `\&` and comma; handles italicized math titles and titleless physics citations. |

3. Normalization and type inference: Converts LaTeX accents to NFC Unicode (`latex_to_text`), formats authors as *"Given Family"* (`parse_free_authors`), and infers the entry type (*phdthesis*, *inproceedings*, *book*, *techreport*, *article*, or *misc*).

| Raw `\bibitem` Input | Extracted Record |
|---|---|
| `\bibitem{GRW} G. C. Ghirardi, A. Rimini, and T. Weber, ``Unified dynamics,'' Phys. Rev. D \textbf{34}, 470 (1986).` | key: *GRW*, type: *article*, authors: *[G. C. Ghirardi, A. Rimini, T. Weber]*, title: *Unified dynamics*, journal: *Phys. Rev. D*, year: *1986* |
| `\bibitem[Aziz(2020)]{Az20} Haris Aziz and J{\'e}r{\^o}me Monnot. \newblock Computing pareto optimal committees. \newblock \emph{Autonomous Agents}, 2020. \newblock \doi{10.1007/s10458-020-09445-y}.` | key: *Az20*, type: *article*, authors: *[Haris Aziz, Jérôme Monnot]*, title: *Computing pareto optimal committees*, journal: *Autonomous Agents*, year: *2020*, doi: *10.1007/s10458-020-09445-y* |
| `\bibitem{EM} M. R. Evans and S. N. Majumdar, Phys. Rev. Lett. \textbf{106}, 160601 (2011).` | key: *EM*, type: *article*, authors: *[M. R. Evans, S. N. Majumdar]*, title: *None* (physics style), journal: *Phys. Rev. Lett*, year: *2011* |

---

### 1.9. Cross-Version Deduplication (Reference Entries, `\cite` Renaming, and Full-Text Elements)

#### Level 1: Reference Deduplication, Field Unionization, and Citation Renaming
`dedup()` groups all `(version, Entry)` records of a publication using a constrained Union-Find structure across four passes:

```{mermaid}
flowchart LR
    P1["Pass 1: Same (version, key)<br/>Force merge within version"] --> P2["Pass 2: Shared DOI or arXiv ID<br/>Merge if no identifier/title conflict"]
    P2 --> P3["Pass 3: Identical normalized title<br/>(≥ 12 chars, ≥ 2 words)<br/>Requires |Δyear| ≤ 1 & shared 1st author"]
    P3 --> P4["Pass 4: Same citation key across versions<br/>Merge unless DOI, arXiv ID, or<br/>title Jaccard (< 0.5) conflicts"]
```

- Conflict vetoes: Merges are blocked when entries have different DOIs or arXiv IDs. In Pass 3 (title match), merges are also vetoed if years differ by more than 1 or first-author surnames disagree (keeping LeCun's 2015 article *"Deep learning"* separate from Goodfellow's 2016 book *"Deep learning"*). In Pass 4 (same key across versions), merges are vetoed if identifiers conflict or title token Jaccard similarity is below 0.5; the older version's key is then disambiguated (e.g., *smith2020_2*).
- Canonical key and field union: Each merged group takes the key from its most recent version. Fields are unioned by sorting the group's entries from newest version to oldest (preferring `.bib` and fuller entries within a version) and selecting the first non-null value per field, synthesizing `https://arxiv.org/abs/{eprint}` when `url` is absent:

| Version | Original Key | Title | Authors | Year | DOI |
|:---:|---|---|---|:---:|---|
| $v_1$ | *vaswani2017* | Attention Is All You Need | [Ashish Vaswani] | 2017 | *None* |
| $v_2$ | *vit_transformer* | Attention is all you need | [Ashish Vaswani] | 2017 | 10.5555/3295222 |
| $v_3$ | *vit_transformer* | Attention is all you need | [Ashish Vaswani] | *None* | *None* |
| Unified | *vit_transformer* | Attention is all you need | [Ashish Vaswani] | 2017 | 10.5555/3295222 |

- Citation renaming (`rename_cites`): Using the per-version rename map (`vaswani2017` $\to$ `vit_transformer` in $v_1$), `rename_cites()` updates all LaTeX, *natbib*, and *biblatex* citation commands (`\cite`, `\citep`, `\citet`, `\parencites`, etc.), collapsing duplicate keys within the same command and normalizing comma spacing.

#### Level 2: Full-Text Element Deduplication Across Versions
After normalization and cleanup, `add_version()` assigns IDs and inserts nodes into the publication store:

```python
if node is root:
    element_id = f"{self.arxiv_id}#root-v{version}"
    content = f"Version {version} root"
else:
    content = node.content.strip()
    key = (node.kind, content)
    ordinal = occurrence_counts[key]
    occurrence_counts[key] += 1
    element_id = f"{self.arxiv_id}#{node.kind}-{_signature(node.kind, content, ordinal)}"
```

`_signature(kind, content, ordinal)` hashes `f"{kind}\0{content}\0{ordinal}"` with SHA-256 (first 20 hex chars), where `ordinal` counts occurrences of `(kind, content)` within the current version. Because the counter resets per version, the first occurrence ($\text{ordinal} = 0$) of an unchanged sentence in $v_1$ and $v_2$ gets the same `element_id` and is stored once in `elements`, while `hierarchy` stores each version's parent-child edges. Meanwhile, two identical sentences or headings inside the *same* version receive $\text{ordinal} = 0$ and $\text{ordinal} = 1$, keeping their tree nodes distinct.

---

### 1.10. Target Schema Mapping (`elements`, `hierarchy`, and `bibtex` tables)

Each publication is written atomically (via `.tmp` files and `os.replace`) to its own output folder, paired with corpus-level summary files:

```text
m2_processed/
├── 2605-27338/
│   ├── elements.jsonl      # Deduplicated nodes across all versions
│   ├── hierarchy.jsonl     # Per-version (element_id -> parent_id) edges
│   ├── bibtex.jsonl        # Unified, deduplicated BibTeX records
│   ├── stats.json          # Per-paper coverage, gather & dedup statistics
│   └── _SUCCESS            # Completion marker
├── m2_stats.json           # Corpus-level summary & benchmark metrics
├── paper_stats.jsonl       # Concatenated stats.json records of completed papers
└── errors.jsonl            # Paper-level exceptions, if any
```

On restart, `process_paper()` skips any paper with a valid `_SUCCESS` marker and matching `PARSER_SCHEMA_VERSION`, reusing its cached `stats.json` for the corpus summary.

The three per-paper JSONL files map 1-to-1 to the published Parquet dataset schemas:

| Table | Column | Type | Description |
|---|---|---|---|
| `elements` | `element_id` | string (non-null) | `{arxiv_id}#root-v{n}` for roots, or `{arxiv_id}#{kind}-{sha256[:20]}`. |
| | `arxiv_id` | string (non-null) | Hyphenated publication ID (e.g., 2605-27338). |
| | `kind` | string (dictionary) | *document*, heading rank, *abstract*, *list*, *item*, *sentence*, *formula*, or *figure*. |
| | `content` | string | Cleaned and normalized element text. |
| `hierarchy` | `arxiv_id` | string (non-null) | Hyphenated publication ID. |
| | `version` | int (non-null) | Revision number ($1, 2, \dots$). |
| | `element_id` | string (non-null) | Child node ID in `elements`. |
| | `parent_id` | string (non-null) | Parent node ID in `elements`. |
| `bibtex` | `arxiv_id` | string (non-null) | Hyphenated citing publication ID. |
| | `citation_key` | string (non-null) | Canonical citation key after deduplication. |
| | `entry_type` | string (dictionary) | *article*, *inproceedings*, *book*, *phdthesis*, *techreport*, or *misc*. |
| | `title` | string (nullable) | Plain-Unicode title of the cited work. |
| | `authors` | list of strings | Author names in *"Given Family"* order. |
| | `year` | int (nullable) | 4-digit publication year. |
| | `doi`, `journal`, `url` | string (nullable) | Normalized DOI, venue name, and URL (or synthesized arXiv abstract link). |

```{mermaid}
erDiagram
    PAPERS ||--o{ ELEMENTS : "contains deduplicated"
    PAPERS ||--o{ HIERARCHY : "has version edges"
    PAPERS ||--o{ BIBTEX : "cites unified"
    ELEMENTS ||--o{ HIERARCHY : "referenced as element_id / parent_id"

    ELEMENTS {
        string element_id PK
        string arxiv_id FK
        string kind
        string content
    }

    HIERARCHY {
        string arxiv_id FK
        int version
        string element_id FK
        string parent_id FK
    }

    BIBTEX {
        string arxiv_id FK
        string citation_key PK
        string entry_type
        string title
        list authors
        int year
        string doi
        string journal
        string url
    }
```

---

## 2. Results and Performance Evaluation

### 2.1 Parsing Coverage and Version-Level Success Rate

We report coverage at two levels. A **version** counts as *parsed* when the parser found a compilation root, flattened the `\input`/`\include` path and built a hierarchy tree. A version without usable TeX source is kept as an *empty* record rather than skipped, as the milestone requires, so the statistics reflect the real corpus. A **paper** is *fully parsed* when all its versions were parsed, *partially parsed* when only some were, and *not parsed* when none were.

```{code-cell} ipython3
:tags: [remove-input]
import sys
from pathlib import Path
for _p in (Path.cwd(), Path.cwd().parent):
    if (_p / "analysis").is_dir() and str(_p / "analysis") not in sys.path:
        sys.path.insert(0, str(_p / "analysis"))

import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from IPython.display import display
import m2_analysis as m2

S2 = m2.load_all("../parsing_stats")
T2 = m2.tables(S2)
```

```{code-cell} ipython3
:tags: [remove-input]
display(T2["coverage"])
m2.plot_coverage(S2); plt.show()
```
All 25,000 publications were processed without a parser failure (0 errors). At version level, 32,652 of 34,373 versions (94.99%) were parsed and 1,721 (5.01%) were empty. The empty share is stable across version indices, at roughly 5% for v1 and 3–4% for later versions, so late versions are not harder to parse than early ones. At paper level, about 16.7k of the 17.8k single-version
papers were parsed. Among papers with several versions, most are fully parsed (for example 5.4k of 5.7k two-version papers), and only a small number are partially parsed (97 for two versions, 32 for three).

**Interpretation.** The 5% gap does not come from parsing errors. It comes from versions that Milestone 1 delivered without a usable TeX source. These are recorded as empty, which includes 3 versions whose source folder is missing. The paper-level panel supports this: a paper's versions tend to be empty together, so for multi-version papers "no version parsed" is much more common
than "partially parsed". The parse success rate of 100% only says that no paper crashed the parser, so version-level coverage is the more informative metric.

```{note}
Both panels use a logarithmic y-axis so that small categories remain visible; the exact counts are in the table above.
```

### 2.2 Element and Structural Distribution

```{code-cell} ipython3
:tags: [remove-input]
m2.plot_kinds(S2); plt.show()
```
Before deduplication, the parser extracted 12,586,956 elements across 32,652 parsed versions. `Sentences` account for the largest share, with 9.84 million elements (78.2%), averaging around 300 sentences per version. Block `formulas` come next, making up 1.21 million elements (9.6%). Together with `figures` (254k, 2.0%), as specified in the milestone, tables are grouped under the `figure` kind, these three leaf kinds required by the milestone represent 87.8% of all extracted elements. The remaining 12.2% consist of structural nodes. Each parsed version has exactly one `document` root, while 30,484 versions (0.2%) contain an `abstract`.

The distribution also gives a rough picture of how scientific papers are structured. On average, each version contains about 7 `sections`, 9 `subsections`, and 7 `paragraph-level headings`. Lists have around 3 `items` each. There is approximately one block formula for every 8 sentences. However, this does not capture all mathematical content, since inline equations are included within sentences rather than counted separately.

Some heading levels appear much less frequently, such as `chapter` (634), `part` (132), and `subparagraph` (807). These are likely found in documents with less common or more complex heading structures, including theses, books, and articles with deeply nested sections.

```{note}
The x-axis uses a logarithmic scale because the number of sentences is nearly five orders of magnitude greater than that of the rarest element kinds. This makes the smaller categories easier to distinguish in the plot, while the table above provides the exact counts.
```



```{code-cell} ipython3
:tags: [remove-input]
m2.plot_kind_mix(S2); plt.show()
```
Grouping the 23,697 parsed papers by their maximum hierarchy depth shows how the element distribution changes across papers with different structures. `Sentences` make up the largest share in every group, but their proportion gradually drops from 84% in papers with a depth of 2 or less to 76% in papers with a depth of 6 or more. Block `formulas` remain fairly consistent, accounting for around 8–11% across the groups.

In deeper papers, structural elements make up a larger share of the total. This is especially clear for `list items`, `paragraph`, `subsections`, and `lists`. This is expected, as papers with more hierarchy levels tend to have more headings and nested structures. Depth 4 is the largest group, with 7,003 papers. Meanwhile, 5,962 papers (25%) have a maximum depth of 6 or more.


```{code-cell} ipython3
:tags: [remove-input]
m2.plot_depth(S2); plt.show()
```
The hierarchical distribution shows that most elements are concentrated at depths 2–4, with depth 3 containing the largest number of elements (5.49 million). The number of elements decreases substantially at deeper levels, indicating that deeply nested structures are less common. Similarly, most papers have a maximum hierarchy depth between 3 and 6, with depth 4 being the most frequent (approximately 7,000 papers). Only a small number of papers reach depths above 9. Overall, the results suggest that most parsed papers have moderately deep hierarchical structures, while highly nested documents are relatively rare.

### 2.3 Deduplication Efficiency

```{code-cell} ipython3
:tags: [remove-input]
display(T2["dedup"])
m2.plot_dedup(S2); plt.show()
```

Deduplication reduced the total number of elements from 12.59 million to 9.83 million (21.9%), while BibTeX entries decreased from 1.34 million to 0.98 million (26.4%). The reduction becomes more significant as the number of parsed versions per paper increases. For papers with 2 versions, the reduction reaches 37.9% for elements and 45.7% for BibTeX entries, rising to 63.2% and 74.5%, respectively, for papers with 5 or more versions. This indicates that deduplication is particularly effective for papers with multiple versions, where repeated elements and references are more likely to occur.

### 2.4 Qualitative Examples of Parsed Trees and Unified BibTeX Records

**Example of a Parsed Hierarchical Tree**

```
Document
│
├── Abstract
│   ├── Sentence
│   ├── Sentence
│   └── Sentence
│
├── Section: Introduction
│   ├── Paragraph
│   │   ├── Sentence
│   │   └── Sentence
│   └── Subsection: Background
│       ├── Sentence
│       └── Formula
│
├── Section: Methodology
│   ├── Subsection: Proposed Approach
│   │   ├── Paragraph
│   │   │   ├── Sentence
│   │   │   └── Formula
│   │   └── Figure
│   └── Subsection: Implementation
│       ├── Sentence
│       └── List
│           ├── Item
│           └── Item
│
├── Section: Results
│   ├── Figure
│   ├── Formula
│   └── Sentence
│
└── Section: Conclusion
    ├── Sentence
    └── Sentence
```

**Example of a Unified BibTeX Record**

```
# Version 1
@article{reference_v1,
  title = {Example Research Paper},
  author = {Author A and Author B},
  year = {2020}
}

# Version 2
@article{reference_v2,
  title = {Example Research Paper},
  author = {Author A and Author B},
  doi = {10.xxxx/example}
}
```

### 2.5 Parser Performance

```{code-cell} ipython3
:tags: [remove-input]
display(T2["performance"])
m2.plot_performance(S2); plt.show()
```
The complete corpus of 25,000 papers was processed using 8 workers in 578.6 seconds (approximately 9.64 minutes), achieving a throughput of 43.21 papers per second.

The wall-clock time averaged 0.023 seconds per paper, while the reported average processing time was 0.184 seconds per paper. The difference is due to parallel execution across multiple workers.

Regarding memory consumption, the parser used an average of 454.0 MB of RAM, with a peak usage of 696.7 MB. The final output occupied 4.80 GB, equivalent to approximately 201.3 KB per paper.