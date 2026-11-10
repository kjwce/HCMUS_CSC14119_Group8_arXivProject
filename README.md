# HCMUS CSC14119 Group 8 — arXiv project code

This repository keeps the source code of Milestones 1 and 2 separately. Raw and
processed corpora stay outside GitHub.

## Repository layout

- `src/m1/scraper_m1.py`: resumable arXiv and Semantic Scholar crawler.
- `src/m2/models.py`: shared `Node` contract.
- `src/m2/gather.py`: root detection and transitive multi-file gathering.
- `src/m2/hierarchy.py`: Document/heading/list/sentence/formula/figure tree builder.
- `src/m2/normalize.py`: sentence splitting, cleanup and math/citation normalization.
- `src/m2/bib.py`: `.bib`/`\\bibitem` extraction and cross-version deduplication.
- `src/m2/store.py`: cross-version content deduplication and the `elements`,
  `hierarchy`, and `bibtex` working tables.
- `src/m2/run_m2.py`: resumable multiprocessing runner and corpus inventory command.
- `tests/`: unit and integration-contract tests for the M2 modules.

## Environment

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run Milestone 1

Set `S2_API_KEY` only when Semantic Scholar access is needed. The arXiv source
download itself does not require a key.

```powershell
$env:S2_API_KEY = "your-key"
python -m src.m1.scraper_m1 --start 2603.05580 --end 2603.10579 --out "D:\path\to\m1_data"
```

Do not commit `s2_api_key.txt` or the output directory.

## Validate the M1 corpus for Milestone 2

```powershell
python -m src.m2.run_m2 `
  --input "D:\path\to\m1_raw" `
  --inventory
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The equivalent shorter command is `python -m pytest -q`.

## Run the integrated parser

```powershell
python -m src.m2.run_m2 `
  --input "D:\path\to\raw" `
  --output "D:\path\to\m2_processed" `
  --workers 8
```

Use `--limit 20` for a sample run. A `_SUCCESS` marker makes later runs skip
completed publications; pass `--force` to rebuild them.

Add `--archive` to also pack the output folder into `m2_processed.tar.gz`
(next to the folder) when the run ends, ready to upload to Drive. Use
`--archive "D:\path\to\m2_all.tar.gz"` to choose the file name. The folder is
kept so a later run can still resume from it.

The corpus-level `m2_stats.json` records paper success, version coverage,
deduplication counts, unresolved citation keys, running time, RAM and output
size. `paper_stats.jsonl` collects the complete `stats.json` record from every
successfully processed paper into one streaming-friendly file. Resume-only runs
preserve the last complete fresh-run benchmark and put their own timing under
`last_run`. Each paper's `stats.json` also records its compilation root,
used/unused files and include diagnostics.

Each publication is written independently:

```text
m2_processed/<arxiv-id>/
├── elements.jsonl
├── hierarchy.jsonl
├── bibtex.jsonl
├── stats.json
└── _SUCCESS
```

At the output root, `m2_stats.json`, `paper_stats.jsonl`, and `errors.jsonl`
provide the corpus summary, all per-paper diagnostics, and failures respectively.

Raw data, logs, archives, secrets, and generated outputs are excluded by
`.gitignore` and must not be pushed to GitHub.

## Milestone 3 — Reference Matching

> Status: skeleton. Sections marked `TODO (owner)` are filled in as the code and
> results land.

Goal: for every BibTeX entry of a paper (M2 `bibtex` table), find the crawled
reference of the same paper (M1 `references` table, arXiv-keyed only) that it
cites, or decide that none matches. Each entry has at most one correct
reference. The system ranks candidates and is evaluated with MRR@5.

### Pipeline

```text
bibtex (M2) ──┐
              ├─► 1. Cleaning ─► 2. Labelling ─► 3. Features ─► 4. Model ─► 5. MRR@5
references(M1)┘                                                       │
                                                                      ▼
                                              predictions table + Hugging Face dataset
```

1. **Cleaning**: TODO (Hưng) — title/author/year normalisation, arXiv ID and DOI
   extraction from BibTeX fields.
2. **Labelling**: manual labels for 25 papers plus automatic labels from a
   rule-based matcher (see *Label format*).
3. **Features**: one feature vector per (entry, reference) pair of the same paper
   (see *Features*).
4. **Model**: TODO (Ngọc) — problem framing and model choice.
5. **Evaluation**: top-5 candidates per labelled entry, MRR@5 on the test set.

Whether LLMs or pretrained models are used: TODO (team decision; the course
grades such groups separately).

### How to run Milestone 3

TODO (Hưng, Ngọc): commands for each step, inputs and outputs.

```powershell
# python -m src.m3.<module> --input ... --output ...
```

### Label format

TODO (Đức): review and complete.

Manual labels are stored per member range in `rX_manual_entries.csv`
(UTF-8 CSV, one row per BibTeX entry). Candidate references for lookup are in
`rX_manual_references.csv`.

| Column | Meaning |
|---|---|
| `range`, `arxiv_id`, `citation_key` | Identify the entry; `arxiv_id` uses the dotted form `2605.30939` |
| `bib_title`, `bib_authors`, `bib_year`, `bib_doi`, `bib_journal`, `bib_url` | Entry fields copied from the M2 `bibtex` table (authors joined by ` \| `) |
| `match_status` | `matched` or `no_match` |
| `groundtruth_arxiv_id` | The `ref_arxiv_id` of the true reference when `matched`, empty otherwise |
| `annotator_notes` | Free text; `CẦN ĐỨC KIỂM TRA` marks rows awaiting review |

Labelling rules:

- Candidates come only from the references of the same paper.
- Evidence, strongest first: shared arXiv ID or DOI, same title (ignoring case,
  punctuation and LaTeX), matching authors, year (±1 between preprint and
  published version), venue.
- `no_match` is a valid label: books, websites, works not on arXiv, or works
  Semantic Scholar did not return.
- Each manually labelled paper has at least the first 20 entries labelled in
  their original order, so easy entries are not cherry-picked.

Automatic labels: TODO (Hưng) — matcher rules, output file and format.

### Labelling statistics

| Pool | Papers | Labelled entries | matched | no_match |
|---|---|---|---|---|
| Manual, range 1 | 5 | 193 | 135 | 58 |
| Manual, ranges 2–5 | TODO | TODO | TODO | TODO |
| Automatic | TODO (≥ 2,498) | TODO | TODO | TODO |

### Data split

Split by paper with a fixed seed: test = 5 manual + 5 automatic papers,
validation = 5 manual + 5 automatic papers, train = all remaining labelled
papers. TODO (Hưng): seed, paper lists and entry counts per split.

### Features

TODO (Kiệt, with Ngọc's plots): one row per feature, each backed by a plot or
number from the labelled data.

| Feature | Idea | Evidence from the data |
|---|---|---|
| TODO | TODO | TODO |

### Results and analysis

TODO (Kiệt): MRR@5 on test (manual vs automatic papers), validation score,
feature importance, error analysis.

Observation from manual labelling (range 1): several entries labelled
`no_match` carry an arXiv link in the BibTeX itself (for example `silva2022` →
2206.02826, `healthllm` → 2401.06866), but that paper is missing from the
Semantic Scholar references. These are crawl-side misses rather than model
errors, and they bound the reachable MRR@5.

### Hugging Face dataset

TODO (Vân Anh, Hưng): dataset link and the six configs (`papers`,
`references`, `elements`, `hierarchy`, `bibtex`, `predictions`).
