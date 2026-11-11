# HCMUS CSC14119 Group 8 — arXiv project code

This repository keeps the source code of Milestones 1, 2, and 3 separately. Raw and
processed corpora stay outside GitHub.

Published dataset: [Group 8 arXiv corpus on Hugging Face](https://huggingface.co/datasets/hngx1016/HCMUS_CSC14119_Group8_arxiv).
The six configurations contain the corpus tables and matching predictions;
the dataset card includes the model selection and evaluation reports.

## Start here

Run all commands from the repository root. Examples below use **PowerShell**;
replace the example `D:\path\to\...` and `D:\m3_work\...` paths with your own
data locations. No path in this guide requires a particular member's computer.

| Stage | Input | Main output |
| --- | --- | --- |
| M1: crawl | Assigned arXiv ID range; Semantic Scholar access | `raw/<paper>/` sources, metadata and references; range manifest |
| M2: parse | Extracted M1 `raw/` | Per-paper elements, hierarchy, BibTeX and parser statistics |
| M3: prepare | Full M1/M2 corpus, five range manifests and reviewed labels | Frozen labels, publication partitions and flat modelling inputs |
| M3: model | Frozen modelling handoff | Features, selected model, metrics, top-five predictions and plots |
| M3: publish | Full corpus plus actual model predictions | Six validated Parquet configurations on Hugging Face |

For a fresh full run, follow the M1, M2, matching-data, modelling and Parquet
sections **in that order**. If M1/M2 are already complete, reuse those outputs;
there is no need to crawl again. If a reviewed modelling handoff is already
available, start at **Generate features** using that handoff unchanged.

To inspect the published dataset, use the Hugging Face link above; training and
API credentials are not required just to view it. To reproduce the exact
experiment, obtain the group's frozen `handoff_ngoc_v2` package or its reviewed
CSV inputs and original automatic pool. These working inputs are not included
in the GitHub source repository. A new shortlist or new labels constitute a new
experiment and may yield different results.

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
- `src/m3/data.py`: strict M1-reference/M2-BibTeX readers and corpus inventory.
- `src/m3/clean.py`: deterministic title, author, DOI, and arXiv-ID cleaning.
- `src/m3/matcher.py`: auditable classic automatic labeller with per-field scores.
- `src/m3/selection.py`: deterministic range-balanced manual-label shortlist.
- `src/m3/audit.py`: metadata audit and final manual-pool recommendations.
- `src/m3/split.py`: leakage-safe publication-level train/valid/test splitting.
- `src/m3/run_m3.py`: M3 inventory, automatic-labelling, and split CLI.
- `src/m3/manual_handoff.py`: reviewed-label import and frozen modelling inputs.
- `src/m3/feature_engineering.py`: train-only TF-IDF and bounded feature export.
- `src/m3/train_model.py`: classical model fitting and validation selection.
- `src/m3/evaluate.py`, `src/m3/prediction.py`: MRR evaluation and top-five ranking.
- `src/m3/export_parquet.py`: six-table export, validation and dataset card.
- `tests/`: unit and integration-contract tests for the M1-M3 modules.

## Environment

Python 3.11 or 3.12 is recommended; the reported model run used Python 3.12.7.
Clone the shared `main` branch first:

```powershell
git clone --branch main https://github.com/kjwce/HCMUS_CSC14119_Group8_arXivProject.git
cd HCMUS_CSC14119_Group8_arXivProject
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in subsequent commands. Keep generated data outside this repository.
M2/M3 run on CPU; a GPU is not required. Start with two workers for M2 and
increase only if the machine has enough memory.

Check the installation from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The equivalent shorter command is `python -m pytest -q`. The integrated code
passed 97 tests when the current workflow was checked.

## Run Milestone 1

Set `S2_API_KEY` only when Semantic Scholar access is needed. The arXiv source
download itself does not require a key.

```powershell
$env:S2_API_KEY = "your-key"
python -m src.m1.scraper_m1 --start 2603.05580 --end 2603.10579 --out "D:\path\to\m1_data"
```

Do not commit `s2_api_key.txt` or the output directory.

The example range is one member's assignment, not all 25,000 group papers.
Each member supplies their assigned range. `--limit 20` is suitable for a smoke
run; omit it for the full assignment. `--no-s2` skips reference collection and
therefore does not produce the complete M1 input required for M3 matching.

M1 output has the following structure:

```text
m1_data/
  raw/
    2603-05580/
      metadata.json
      references.json
      v1/                 # downloaded source files for this version
      v2/                 # present only if this revision exists
  manifest/papers.jsonl   # keep the manifest of each member's range
  logs/
```

If input arrives as `.zip` or `.tar.gz`, extract it first. M2/M3 expect folders,
not an archive path. For the group run, combine the **publication folders** from
the five extracted `raw/` directories into one `m1_all/raw/`; do not put five
range-level wrappers inside it. Check duplicate IDs before combining files.
Keep each range's `manifest/papers.jsonl` separately for the manual shortlist.

```text
m1_all/raw/<paper>/metadata.json
m1_all/raw/<paper>/references.json
m1_all/raw/<paper>/v1/...
```

## Run Milestone 2

### Validate the M1 corpus

```powershell
python -m src.m2.run_m2 `
  --input "D:\path\to\m1_all\raw" `
  --inventory
```

### Parse the sources

```powershell
python -m src.m2.run_m2 `
  --input "D:\path\to\m1_all\raw" `
  --output "D:\path\to\m2_schema3" `
  --workers 2
```

Use `--limit 20` for a sample run. A `_SUCCESS` marker makes later runs skip
completed publications; pass `--force` to rebuild them.

Add `--archive` to also pack the output folder into `m2_schema3.tar.gz`
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
m2_schema3/<arxiv-id>/
├── elements.jsonl
├── hierarchy.jsonl
├── bibtex.jsonl
├── stats.json
└── _SUCCESS
```

At the output root, `m2_stats.json`, `paper_stats.jsonl`, and `errors.jsonl`
provide the corpus summary, all per-paper diagnostics, and failures respectively.
An empty `errors.jsonl` means no failures were recorded. Check
`parser_schema_version: 3` in `m2_stats.json` before using this output for M3.

Raw data, logs, archives, secrets, and generated outputs are excluded by
`.gitignore` and must not be pushed to GitHub.

## Prepare Milestone 3 matching data

M3 joins each publication's unified `bibtex.jsonl` from M2 with only the
arXiv-keyed rows of its M1 `references.json`. Modern arXiv IDs are converted to
their public dotted form and version suffixes are removed at this boundary. The
CLI requires M2 parser schema 3 by default, preventing an older full-corpus run
from being used accidentally.

The order in this section is inventory → shortlist → audit → finalize manual
pool → annotation package → automatic labels → reviewed handoff. The handoff
command freezes the final partitions and calibrates the automatic matcher on
manual Train only. It is the supported path for the reported experiment.

### 1. Inventory and shortlist

Audit the complete input before selecting any labelled publications:

```powershell
python -m src.m3.run_m3 inventory `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --out "D:\m3_work\publication_inventory.jsonl"
```

Create ten manageable candidates per member range (five recommended and five
backups) from the inventory and the M1 manifests:

```powershell
python -m src.m3.run_m3 manual-shortlist `
  --inventory "D:\m3_work\publication_inventory.jsonl" `
  --range-manifest "r1=D:\path\to\r1\manifest\papers.jsonl" `
  --range-manifest "r2=D:\path\to\r2\manifest\papers.jsonl" `
  --range-manifest "r3=D:\path\to\r3\manifest\papers.jsonl" `
  --range-manifest "r4=D:\path\to\r4\manifest\papers.jsonl" `
  --range-manifest "r5=D:\path\to\r5\manifest\papers.jsonl" `
  --out "D:\m3_work\manual_shortlist.jsonl"
```

Audit those 50 candidates against the real M1/M2 records and create a
spreadsheet-friendly CSV with five final recommendations per range:

```powershell
python -m src.m3.run_m3 audit-shortlist `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --shortlist "D:\m3_work\manual_shortlist.jsonl" `
  --out "D:\m3_work\manual_shortlist_audited.jsonl" `
  --csv-out "D:\m3_work\manual_shortlist_audited.csv"
```

### 2. Finalize the manual publication pool

Review the audited recommendations with the group. The following command reads
the audited **JSONL**, retains rows with `final_recommended: true`, and writes
five selected publications per range to the file required by later steps:

```powershell
python -m src.m3.run_m3 finalize-manual-pool `
  --audited-shortlist "D:\m3_work\manual_shortlist_audited.jsonl" `
  --per-range 5 `
  --out "D:\m3_work\manual_pool.jsonl"
```

Check that the output reports five ranges and 25 manual publications. Editing
the CSV display alone does not change the JSONL that this command consumes.
Finalize any replacements before generating the automatic pool so the two
pools stay disjoint. The published experiment used the final reviewed papers,
including the approved range-5 replacements, as recorded in its handoff.

Do not tune features or model parameters on the test publications. Generated
inventories, labels, partitions, predictions, and Parquet files are data and
must remain outside this source repository.

### 3. Build and review the annotation files

Create separate entry/reference CSV files for the manual annotators:

```powershell
python -m src.m3.run_m3 build-manual-package `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --manual-pool "D:\m3_work\manual_pool.jsonl" `
  --output-dir "D:\m3_work\manual_label_package"
```

For the supported full-coverage workflow, annotate **every exported entry** of
each selected publication; each paper must have at least 20 reviewed entries.
The minimum of 20 alone does not provide the complete label coverage required
by this feature/training/export pipeline. Set `match_status` to `matched` and
fill `groundtruth_arxiv_id` with one ID
from the corresponding reference file, or set `match_status` to `no_match` and
leave the ground-truth field empty. Candidate references never cross publication
boundaries.

#### Manual label format and evidence

The manual files are UTF-8 CSVs named `rX_manual_entries.csv`; the companion
`rX_manual_references.csv` lists candidates from the same publication.

| Column | Meaning |
| --- | --- |
| `range`, `arxiv_id`, `citation_key` | Entry identity; publication arXiv ID uses its canonical dotted form. |
| `bib_title`, `bib_authors`, `bib_year`, `bib_doi`, `bib_journal`, `bib_url` | M2 entry metadata; authors are joined with a vertical bar. |
| `match_status` | `matched` or `no_match`. |
| `groundtruth_arxiv_id` | Reference ID from the same paper when matched; empty otherwise. |
| `annotator_notes` | Evidence, ambiguity or a note requesting human review. |

Compare identifiers first, then title, authors, year and venue. A year difference
can reflect preprint versus publication dates and should be checked rather than
rejected automatically. Label the supplied entries consistently, without selecting
only easy matches. Preserve identifiers as text when using spreadsheet software.

An entry may contain an arXiv link even though that work is absent from its M1
Semantic Scholar reference candidates. In that case `no_match` refers to the
available candidate pool, not to whether the cited work exists on arXiv. Such
crawl-side omissions limit attainable all-entry MRR@5 independently of ranking.

### 4. Create and validate automatic labels

For 25,000 group publications and a 25-paper manual pool, at least 2,498 other
publications are needed: `ceil((25000 - 25) / 10)`. Automatic output contains
one row per BibTeX entry, including explicit no-match rows. Scores and margins
are retained for auditing. This step can run while manual labels are reviewed,
provided the final manual publication IDs have already been fixed.

```powershell
python -m src.m3.run_m3 auto-label `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --manual-pool "D:\m3_work\manual_pool.jsonl" `
  --target-papers 2498 `
  --pool-out "D:\m3_work\automatic_pool.jsonl" `
  --out "D:\m3_work\automatic_labels.jsonl"

python -m src.m3.run_m3 validate-auto-labels `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --automatic-pool "D:\m3_work\automatic_pool.jsonl" `
  --automatic-labels "D:\m3_work\automatic_labels.jsonl"
```

### 5. Import reviewed labels and freeze the modelling handoff

After the five reviewed range CSVs are available, validate them against M1/M2,
freeze the publication split, and calibrate automatic acceptance thresholds on
manual **train only**. The manual pool is taken from the reviewed submissions,
so an approved replacement of a range's papers is reflected in the new split.
Place the reviewed files in `ok_entries/` with the exact names
`r1_manual_entries.csv` through `r5_manual_entries.csv`. The importer validates
the five files, their M2 entry identities and same-paper reference targets.

```powershell
python -m src.m3.manual_handoff `
  --entries-dir "D:\path\to\ok_entries" `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --automatic-pool "D:\m3_work\automatic_pool.jsonl" `
  --automatic-labels "D:\m3_work\automatic_labels.jsonl" `
  --output "D:\m3_work\handoff_ngoc_v2"
```

Use a new output directory per run. It contains the new manual pool, reviewed
labels, partitions, calibrated automatic labels, split-specific labels, flat
BibTeX/reference model inputs, calibration diagnostics, source checksums and a
Vietnamese guide. No Cartesian candidate matrix is held in memory during export.
The previous raw corpus and reviewed CSVs remain unchanged.

Modern and old arXiv IDs are strings, including leading/trailing zeroes. Invalid
or Excel-truncated ground truths are rejected instead of silently padded.
Only acceptance threshold and margin are calibrated. The search targets at
least 98% manual-Train match precision; if no setting qualifies, the code retains
the baseline and records that fallback in `matcher_calibration.json`. This is
an empirical training criterion, not a guarantee on unseen papers. Holdout
labels are used only for diagnostics. Automatic labels are heuristic silver
labels; the reviewed manual pool includes AI-assisted annotation and human
review, so it is not an independent purely human benchmark.

Use the handoff's `labels.jsonl` and `partitions.jsonl` together for all following
commands. Do not mix them with earlier standalone split files. With seed 14119,
the defaults place five manual and five automatic publications in each holdout;
the remaining publications form Train. The lower-level `split` and `split-pools`
commands remain available through `--help` for other workflows, but are not an
additional step after this handoff.

## Run M3 feature engineering and modelling

The model scores each (BibTeX entry, reference) pair within one publication and
ranks that publication's candidates. All implementation files live in `src/m3/`.
The integrated recipe uses 43 classic features, all available candidates and a
Train-only TF-IDF vocabulary. Generated features, models, metrics and plots stay
outside GitHub. Use the frozen handoff from the previous section throughout.

| Module | Responsibility |
| --- | --- |
| `feature_engineering.py` | Cleaning, Ngoc's feature functions, Train-only TF-IDF, bounded feature export |
| `model_data.py` | Canonical string IDs, frozen split/label validation, many-to-one entry-label merge, batch loading |
| `train_model.py` | Train-only scaling and fitting, Validation model selection, frozen selection before Test |
| `evaluate.py` | Entry-level MRR@5 with separate manual/automatic and matched-only diagnostics |
| `prediction.py` | Rank candidates by learned match probability and export the nine prediction columns |

### Generate features

```powershell
python -m src.m3.feature_engineering prepare `
  --handoff "D:\m3_work\handoff_ngoc_v2" `
  --output "D:\m3_work\features_train_only_v1"
```

The handoff contains flat JSONL. All source/reference IDs remain strings and are
canonicalized to dotted public arXiv notation (old IDs retain their archive and
slash). Labels join on publication `arxiv_id` + `citation_key` with a many-to-one
validation. A null ground truth is a labelled no-match, so every candidate of
that entry has pair label 0. Duplicate labels and unknown entry keys are errors,
not silently dropped records.

TF-IDF uses character n-grams of lengths 3 and 4, float32, at most 50,000
features and a deterministic sample of at most 100,000 Train documents. It is
fit only on Train publication texts and then reused for all partitions.
Candidate generation retains the complete entry × reference matrix inside
each publication; no Top-100 pruning is applied. Rows are written in bounded
blocks to Parquet. `features_manifest.json` records hashes, the permitted Train
publication IDs, all feature names/descriptions, row counts and runtime.

### Feature definitions and rationale

| Feature | Definition and reason |
| --- | --- |
| `title_ratio` | Character similarity between titles handles minor spelling differences. |
| `title_token_set` | Token-set title similarity tolerates word order and extra words. |
| `title_partial` | Partial title similarity detects shortened titles. |
| `journal_vs_title_ratio` | Checks for a reference title shifted into the entry's journal field. |
| `blob_partial`, `blob_token_set` | Compare reference title with combined entry title, journal and authors to tolerate shifted fields. |
| `tfidf_cos`, `tfidf_cos_title` | Train-fitted character TF-IDF cosine with combined entry text and with title only. |
| `title_contain` | Fraction of reference-title tokens appearing in the combined entry text. |
| `title_jaccard` | Size of title-token intersection divided by union. |
| `auth_hit`, `auth_jaccard` | Surname containment and set overlap provide evidence independent of titles. |
| `auth_first` | First reference author's surname appears in the entry. |
| `auth_cnt_diff` | Absolute author-count difference helps detect mismatches and truncated metadata. |
| `year_diff`, `year_eq` | Absolute year difference and exact equality let the model distinguish same-year matches from preprint/publication differences. |
| `year_in_blob` | Reference year occurs in combined entry text or identifier fields. |
| `id_hit` | Normalized reference arXiv ID occurs in entry DOI/URL text. |
| `len_ratio` | Ratio of shorter to longer combined-entry/reference-title character lengths. |
| `bib_title_missing`, `bib_auth_missing`, `bib_year_missing` | Entry missingness flags distinguish absent evidence from a genuine low similarity. |
| `ref_auth_missing`, `ref_year_missing` | Reference missingness flags serve the same purpose. |
| `n_refs` | Candidate-pool size gives context for ranking difficulty. |

For each of `title_token_set`, `blob_partial`, `tfidf_cos`, `title_contain` and
`auth_hit`, three additional within-entry features compare each candidate with
its competitors: `_rank` (1 is best), `_gap` (score minus maximum), and `_z`
(standardized score). These use the current entry's candidate texts, not labels.
Finally `combo_score` = .4 × blob_partial + .3 × tfidf_cos + .2 × title_contain +
.1 × auth_hit is a fixed heuristic baseline; its `_rank` and `_gap` complete the
43-feature set. No pretrained encoder or external LLM feature is used.

The training command generates distribution plots covering all 43 features from
Train pairs (`plots/feature_distributions_*.png`). These plots support analysis
of how match and nonmatch pairs differ; they are not evidence of causation.

### Fit and select models

```powershell
python -m src.m3.train_model `
  --features "D:\m3_work\features_train_only_v1" `
  --handoff "D:\m3_work\handoff_ngoc_v2" `
  --output "D:\m3_work\model_run_v1"
```

Two classical classifiers are compared:

1. Logistic regression fitted incrementally with `SGDClassifier(loss='log_loss')`,
   average weights, 5 epochs, and alpha in {1e-4, 1e-5}. It sees every Train pair.
   StandardScaler is fitted incrementally on Train only. NaN/inf values are
   imputed to zero by a fixed rule; explicit missingness indicators are retained.
   Class weights are calculated from Train counts to balance rare true matches.
2. `HistGradientBoostingClassifier`, 100 boosting iterations, 15 leaf nodes,
   learning rate .08 and no internal early-stopping split. For bounded RAM, it
   uses all Train positive pairs plus a uniform reservoir of at most 200,000
   Train negative pairs. Class weights are recomputed for that sample.

Both use seed 14119. Select the supervised model using the mean of manual and
automatic Validation MRR@5, giving the two label sources equal weight. The fixed
combo baseline is reported for comparison. Selection is saved in `selection.json`
before Test metrics are calculated; Test labels never select models or settings.
No further model tuning based on the final Test scores should be performed.

Outputs include `best_model.joblib`, `selection.json`, `training_report.json`,
`metrics.json`, `predictions.jsonl`, `predictions.parquet`, `test_error_cases.jsonl`
and analysis PNGs. Pickle/joblib model artifacts must be loaded only from trusted
local experiment runs. They are data artifacts, not GitHub source files.

### Evaluation and predictions

Every entry receives up to five candidates ordered by match probability, with
lexicographic reference-ID tie breaking. Candidate lists are padded with nulls
only when a paper has fewer than five candidates. IDs and ground truths retain
their string/null representation, including leading/trailing zeroes.

Primary MRR@5 averages reciprocal rank over **all labelled entries**, so a
no-match entry contributes zero. Matched-only MRR and Hit@1/Hit@5 are reported
separately. Manual and automatic scores are reported separately because the
automatic labels are heuristic silver labels. With no-match entries, an all-entry
MRR below 1 can occur even when every matched entry is ranked first.

```powershell
python -m src.m3.evaluate `
  --predictions "D:\m3_work\model_run_v1\predictions.jsonl" `
  --handoff "D:\m3_work\handoff_ngoc_v2" `
  --output "D:\m3_work\model_run_v1\metrics_rechecked.json"
```

Reproduce rankings from the frozen trained model without fitting again:

```powershell
python -m src.m3.prediction `
  --features "D:\m3_work\features_train_only_v1" `
  --handoff "D:\m3_work\handoff_ngoc_v2" `
  --model "D:\m3_work\model_run_v1\best_model.joblib" `
  --output "D:\m3_work\predictions_reproduced"
```

API references: [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

## Export the final Hugging Face Parquet dataset

The schemas in `src/m3/parquet_schema.py` follow the official
[M1](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/crawling/),
[M2](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/parsing/), and
[M3](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/matching/)
column specifications. Install the updated `requirements.txt` first.

### Step 1: export the five source tables

Use the full 25,000-publication M1/M2 corpus, not the smaller modelling handoff.
Choose a new output directory. Existing outputs are never overwritten.

```powershell
python -m src.m3.export_parquet export `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --output "D:\m3_work\hf_dataset_v1"
```

The exporter streams bounded batches, writes Zstandard-compressed shards, and
reads every shard back to check schemas, counts, duplicate keys and foreign keys.
Defaults are 8,192 rows per batch and 250,000 rows per shard; override with
`--batch-rows` and `--shard-rows`. A `--limit 10` smoke run is useful for testing,
but is explicitly marked as limited and cannot become a final submission.

```text
hf_dataset_v1/
  README.md                 # HF dataset card: one configuration per present table
  schema.json               # six explicit column/type specifications
  export_report.json        # row counts, validation checks, storage, runtime
  _BASE_TABLES_SUCCESS      # five source tables validated; not final completion
  data/
    papers/part-00000.parquet
    references/part-00000.parquet
    elements/part-00000.parquet
    hierarchy/part-00000.parquet
    bibtex/part-00000.parquet
```

Large tables contain multiple `part-xxxxx.parquet` shards. This is six logical
tables in the final dataset, not a requirement to squeeze everything into six
physical files. Until actual model output arrives, the card lists only five
configurations and clearly says that predictions are pending.

All publication/reference IDs are canonical strings with leading/trailing zeroes
preserved. The `references.arxiv_id` is the citing publication and
`references.ref_arxiv_id` is the cited work. References without arXiv IDs are
excluded. Original element IDs keep their M2 spelling, including a hyphenated
publication prefix; do not rewrite those IDs independently of hierarchy edges.
The only list columns are one-level lists of strings (`authors`, `ref_authors`,
`revised_dates`). Nested version metadata and parser diagnostics are not serialized
into table cells. Dictionary fields retain string categories. An entirely null
column still has its declared string/integer type.

### Step 2: add the trained model's predictions

The training command above already produces `model_run_v1/predictions.jsonl`
and `predictions.parquet`. The exporter accepts JSONL, CSV UTF-8 or Parquet
with **exactly** these columns:

```text
arxiv_id, partition, citation_key, groundtruth_arxiv_id,
cand_1, cand_2, cand_3, cand_4, cand_5
```

Every ID is a string; ground truth and candidates may be null. Rank all available
references up to five, padding with nulls only when fewer candidates exist.
Predictions must cover all BibTeX entries of the matching publications and use
the frozen handoff partitions and reviewed/automatic ground truths unchanged.
Do not convert identifiers to floats or re-save CSV through Excel General mode.

```powershell
python -m src.m3.export_parquet add-predictions `
  --dataset "D:\m3_work\hf_dataset_v1" `
  --input "D:\m3_work\model_run_v1\predictions.jsonl" `
  --labels "D:\m3_work\handoff_ngoc_v2\labels.jsonl" `
  --partitions "D:\m3_work\handoff_ngoc_v2\partitions.jsonl"
```

The command rejects unknown/duplicate entries, incomplete coverage, wrong
partitions, changed ground truths, candidates from another publication and
Excel-truncated IDs. It adds `data/predictions/`, revalidates all six tables,
updates the card and writes `_FINAL_DATASET_SUCCESS`. It never invents model
predictions from the label matcher. Existing predictions are not overwritten.

### Step 3: attach model reports and validate

Attach the actual selection and evaluation reports to the completed dataset
card. The command checks that the model run's predictions match the exported
predictions by SHA-256 before copying the reports.

```powershell
python -m src.m3.export_parquet document-model `
  --dataset "D:\m3_work\hf_dataset_v1" `
  --model-run "D:\m3_work\model_run_v1"
```

```powershell
python -m src.m3.export_parquet validate `
  --dataset "D:\m3_work\hf_dataset_v1"
```

Check `ready_for_final_submission: true` in the validation output. The completed
local folder has `data/predictions/`, `_FINAL_DATASET_SUCCESS`, the dataset card,
the three model reports and the other five corpus tables. These commands only
prepare local files; upload is separate.

### Step 4: publish or read the dataset

The group's completed dataset is already published at
[hngx1016/HCMUS_CSC14119_Group8_arxiv](https://huggingface.co/datasets/hngx1016/HCMUS_CSC14119_Group8_arxiv).
Only a maintainer with write access needs the following upload step when
publishing a reviewed update. Install the optional Hub CLI and authenticate
locally; do not put access tokens into README, source files or chat.

```powershell
python -m pip install huggingface_hub
hf auth login
hf upload hngx1016/HCMUS_CSC14119_Group8_arxiv "D:\m3_work\hf_dataset_v1" . `
  --repo-type dataset `
  --include "data/*/*.parquet" `
  --include "README.md" `
  --include "schema.json" `
  --include "export_report.json" `
  --include "matching_metrics.json" `
  --include "model_selection.json" `
  --include "model_training_report.json"
```

The upload includes the six tables and their documentation; local completion
markers and model binaries are excluded by the explicit file patterns. The
[card configurations](https://huggingface.co/docs/hub/datasets-manual-configuration)
let the Viewer and streaming backend find the shards without a loading script.
Do not upload M1 source archives, API keys, local environments or generated
datasets to the GitHub code repository. Vân Anh checks each of the six Viewer
configurations after publication, including identifier zeroes and null values.

The Hub split named `train` in each configuration is a storage convention for
that whole table. The actual matching split is the `predictions.partition`
column. Do not confuse these two meanings.

To read the public dataset without downloading all tables at once:

```powershell
python -m pip install datasets
```

```python
from datasets import load_dataset

repo_id = "hngx1016/HCMUS_CSC14119_Group8_arxiv"
predictions = load_dataset(repo_id, "predictions", split="train", streaming=True)
print(next(iter(predictions)))
```

Replace `predictions` with `papers`, `references`, `elements`, `hierarchy` or
`bibtex` to select another configuration. The published dataset is usable
through the library independently of the browser Viewer service.

## Reported experiment

These numbers describe the frozen reviewed handoff and completed model run
published with this dataset. They are observations, not promises for new data.

### Label pools and partitions

| Pool | Publications | Labelled entries | Matched | No match |
| --- | ---: | ---: | ---: | ---: |
| Reviewed manual | 25 | 869 | 583 | 286 |
| Automatic (silver) | 2,498 | 108,353 | 51,789 | 56,564 |
| Combined | 2,523 | 109,222 | 52,372 | 56,850 |

The publication split uses seed **14119**. Every citing publication belongs to
exactly one partition; all its entries and candidate rows share that partition.

| Partition | Manual papers | Automatic papers | Manual entries | Automatic entries | Total entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 15 | 2,488 | 541 | 107,755 | 108,296 |
| Validation | 5 | 5 | 169 | 454 | 623 |
| Test | 5 | 5 | 159 | 144 | 303 |

Feature preparation produced **3,322,958 candidate pairs**, 43 feature columns
and 14 Parquet shards. It retained every same-publication candidate rather than
pruning to 100 candidates per entry. The completed run wrote **109,222
predictions**, one per labelled entry across all three partitions.

### Model selection and held-out results

Validation selection averages the manual and automatic all-entry MRR@5, giving
the two sources equal weight. This source-balanced objective differs from the
pooled entry-weighted Validation MRR.

| Model | Source-balanced Validation MRR@5 |
| --- | ---: |
| Fixed combo baseline | 0.555750 |
| Logistic SGD, alpha = 0.0001 (selected) | 0.555750 |
| Logistic SGD, alpha = 0.00001 | 0.549110 |
| Histogram gradient boosting | 0.555750 |

The selected trained model is `logistic_sgd_alpha_0.0001`. It ties the fixed
baseline on Validation, so **no improvement over the baseline is claimed**.
The code retains the first trained candidate when objectives tie. Model
selection was saved before Test evaluation.

| Test label source | Entries | Matched | No match | MRR@5, all entries | MRR@5, matched only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reviewed manual | 159 | 102 | 57 | 0.641509 | 1.000000 |
| Automatic | 144 | 83 | 61 | 0.576389 | 1.000000 |
| Combined | 303 | 185 | 118 | 0.610561 | 1.000000 |

Each no-match entry contributes zero to the primary score. All matched Test
entries ranked their ground-truth reference first in this run, which does not
imply that the system solves no-match detection or generalizes perfectly.
The automatic labels come from heuristics, the reviewed manual labels include
AI assistance, and each holdout has only five publications per source.

Model-run timing was 227.053 seconds; feature preparation took 156.831 seconds.
These timings exclude crawling, parsing, full-corpus Parquet export and upload.
The recorded model environment was Python 3.12.7, scikit-learn 1.9.1 and NumPy
2.5.4. Dependency ranges allow other versions, so consult the reports when
comparing runtimes or reproducing exact numeric results.

Full reports are available in the dataset's
[Files tab](https://huggingface.co/datasets/hngx1016/HCMUS_CSC14119_Group8_arxiv/tree/main):
`model_selection.json`, `matching_metrics.json`, `model_training_report.json`
and `export_report.json`. Locally, the model output also contains feature
distribution plots, validation comparisons and the Test error-case file. An
empty Test error-case file means no matched-entry top-one errors were found;
it does not certify missing-reference labels.

### Published table sizes

| Configuration | Rows |
| --- | ---: |
| `papers` | 25,000 |
| `references` | 456,380 |
| `elements` | 9,826,217 |
| `hierarchy` | 12,554,304 |
| `bibtex` | 984,731 |
| `predictions` | 109,222 |

The five source tables cover the full assigned corpus; predictions cover the
2,523 matching publications. The dataset contains 99 Parquet shards totaling
977,422,919 bytes, plus its documentation and reports.

## Common issues

- **`No module named src`**: change to the cloned repository root before running
  `python -m src...`; activate the environment or use its Python executable.
- **Missing `manual_pool.jsonl`**: run `finalize-manual-pool` after the shortlist
  audit, or use the group's already frozen pool when reproducing its experiment.
- **M2 schema mismatch**: use parser schema 3. For an older corpus, rebuild M2
  with the current code into a new output directory before preparing M3 inputs.
- **Identifier lost a trailing zero in Excel**: re-import the original CSV with
  ID columns typed as text. Changing cell formatting after truncation does not
  restore the lost value; use the original records to correct it.
- **Existing output directory**: the M3 handoff, feature, training and export
  commands protect existing artifacts. Choose a new directory for a new run;
  M2 has its own `_SUCCESS`-based resume behavior.
- **Not enough RAM**: use the documented batched feature/training commands.
  Avoid `read_pairs()` on the full dataset or joining the entire Cartesian
  matrix into one DataFrame.
- **Colab runtime reset**: files under `/content` are temporary. Persist inputs
  and completed outputs to Drive or download them; a saved notebook alone does
  not preserve the runtime files. Use Linux paths and `!python -m ...` for CLI
  calls instead of copying PowerShell backtick continuations into Python cells.
- **Viewer is unavailable**: check the Files tab and try public streaming first.
  A Viewer service error alone does not establish that the Parquet files are
  corrupt; compare against the exporter validation report and config definitions.
