# HCMUS CSC14119 Group 8 — arXiv project code

This repository keeps the source code of Milestones 1, 2, and 3 separately. Raw and
processed corpora stay outside GitHub.

Published dataset: [Group 8 arXiv corpus on Hugging Face](https://huggingface.co/datasets/hngx1016/HCMUS_CSC14119_Group8_arxiv).
The six configurations contain the corpus tables and matching predictions;
the dataset card includes the model selection and evaluation reports.

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

## Prepare Milestone 3 matching data

M3 joins each publication's unified `bibtex.jsonl` from M2 with only the
arXiv-keyed rows of its M1 `references.json`. Modern arXiv IDs are converted to
their public dotted form and version suffixes are removed at this boundary. The
CLI requires M2 parser schema 3 by default, preventing an older full-corpus run
from being used accidentally.

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

After the manual publication pool is fixed, create classic automatic labels for
the required number of publications outside that pool. The output contains one
row for every BibTeX entry, including explicit no-match rows. Scores and the
acceptance margin are retained so label quality can be audited.

```powershell
python -m src.m3.run_m3 auto-label `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --manual-pool "D:\m3_work\manual_pool.jsonl" `
  --target-papers 2498 `
  --pool-out "D:\m3_work\automatic_pool.jsonl" `
  --out "D:\m3_work\automatic_labels.jsonl"
```

Only after the manual labels have been reviewed should the publication-level
split be frozen. For a five-member group, the defaults place five manual and
five automatic publications in each of validation and test.

```powershell
python -m src.m3.run_m3 split `
  --manual-labels "D:\m3_work\manual_labels.jsonl" `
  --automatic-labels "D:\m3_work\automatic_labels.jsonl" `
  --out "D:\m3_work\partitions.jsonl"
```

Do not tune features or model parameters on the test publications. Generated
inventories, labels, partitions, predictions, and Parquet files are data and
must remain outside this source repository.

Validate the automatic output before it is handed to the modelling stage:

```powershell
python -m src.m3.run_m3 validate-auto-labels `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --automatic-pool "D:\m3_work\automatic_pool.jsonl" `
  --automatic-labels "D:\m3_work\automatic_labels.jsonl"
```

Create separate entry/reference CSV files for the manual annotators:

```powershell
python -m src.m3.run_m3 build-manual-package `
  --m1 "D:\path\to\m1_all\raw" `
  --m2 "D:\path\to\m2_schema3" `
  --manual-pool "D:\m3_work\manual_pool.jsonl" `
  --output-dir "D:\m3_work\manual_label_package"
```

For every selected publication, annotate at least 20 rows in its range entry
file. Set `match_status` to `matched` and fill `groundtruth_arxiv_id` with one ID
from the corresponding reference file, or set `match_status` to `no_match` and
leave the ground-truth field empty. Candidate references never cross publication
boundaries.

### Import reviewed labels and prepare the modelling handoff

After the five reviewed range CSVs are available, validate them against M1/M2,
freeze the publication split, and calibrate automatic acceptance thresholds on
manual **train only**. The manual pool is taken from the reviewed submissions,
so an approved replacement of a range's papers is reflected in the new split.

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
Only acceptance threshold and margin are calibrated, with training match
precision constrained to at least 98%; this is an empirical training criterion,
not a guarantee on unseen papers. Holdout labels are used only for diagnostics.
Automatic labels are heuristic silver labels. Record any AI assistance used in
the reviewed manual labels and retain human review for an independent benchmark.

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

### Step 2: add Ngoc's actual predictions

Ngoc supplies JSONL, CSV UTF-8, or a Parquet file with **exactly** these columns:

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
  --input "D:\path\to\predictions.jsonl" `
  --labels "D:\m3_work\handoff_ngoc_v2\labels.jsonl" `
  --partitions "D:\m3_work\handoff_ngoc_v2\partitions.jsonl"
```

The command rejects unknown/duplicate entries, incomplete coverage, wrong
partitions, changed ground truths, candidates from another publication and
Excel-truncated IDs. It adds `data/predictions/`, revalidates all six tables,
updates the card and writes `_FINAL_DATASET_SUCCESS`. It never invents model
predictions from the label matcher. Existing predictions are not overwritten.

### Step 3: validate and publish

```powershell
python -m src.m3.export_parquet validate `
  --dataset "D:\m3_work\hf_dataset_v1"
```

Publish only after the report says `ready_for_final_submission: true`, model
details have been added to the dataset card, and the group has confirmed the
target public Hugging Face repository and the instructor's hosting instructions.
Upload the dataset card and `data/` directory to that dataset repository; the
[card configurations](https://huggingface.co/docs/hub/datasets-manual-configuration)
let the Viewer and streaming backend find the shards without a loading script.
Do not upload M1 source archives, API keys, local environments or generated
datasets to the GitHub code repository. Vân Anh checks each of the six Viewer
configurations after publication, including identifier zeroes and null values.

The Hub split named `train` in each configuration is a storage convention for
that whole table. The actual matching split is the `predictions.partition`
column. Do not confuse these two meanings.

## Integrated modelling pipeline (Ngoc's feature work + shared M3 data)

All implementation files live together in `src/m3/`. Generated features, fitted
models, metrics and plots stay outside the source repository. The feature recipe
was integrated from Ngoc's submitted `feature_engineering.py`; her original input
folder is retained separately. The supported final workflow below rebuilds the
43 classic features on the matching pool with all candidates and a Train-only
TF-IDF vocabulary. The earlier all-corpus features and `tfidf.pkl` are not used
for the reported final experiment because their fit split was not recorded.

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

Use the earlier `export_parquet add-predictions` command with these real model
outputs to finish the sixth dataset table. The full pipeline's label provenance
and small holdout limitations still apply; semantic correctness is not proven
by schema tests or high silver-label agreement.

Attach the actual model selection and evaluation reports to the completed local
dataset card. This checks that the run's predictions match the exported file:

```powershell
python -m src.m3.export_parquet document-model `
  --dataset "D:\m3_work\hf_dataset_v1" `
  --model-run "D:\m3_work\model_run_v1"
```

This prepares local files only. Publishing the dataset to Hugging Face is a
separate step using the group's chosen account and repository.

API references: [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
