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
