# 0. Introduction

## 0.1. Project background

The growth in usage of scientific repositories has transformed modern science communication, with arXiv acting as one of the foremost servers for scientific publications. While harvesting paper metadata (e.g. titles, author lists, abstracts,...) from arXiv is a common enough project/exercise in data engineering, this project aims to achieve more than that: instead of harvesting just the metadata, we will also harvest the source LaTex files (.tex, .bib) as well. 

While scraping the contents of arXiv papers is decidedly not a novel project idea, existing document processing pipelines generally rely on extracting PDF files of said arXiv papers instead of the LaTeX files. However, this presents a problem as PDFs are designed for visual page rendering rather than data extraction, therefore, compiling a document into a PDF has some drawbacks, of which some of the most relevant to our concern are: the loss of explicit citation keys (\cite{...}) and bibliography structure (\bibitem / BibTeX fields). Of course, PDF files have some advantages over LaTeX files, namely, ease of reading (ready to be read as-is, while .tex/.bib files are just raw source code), and consistency (raw LaTeX files are inconsistent in prose, formatting,... among different authors, even revisions,...; this will be elaborated upon further later).  But, for the purpose of this project (specifically, in the context of building a Reference matching System later on in the project), accuracy regarding paper structure and its citation keys, which the compiled PDF files lack, is needed.

In contrast, scraping and processing these original TeX sources instead of the compiled PDFs allows us to preserve the author's exact structural hierarchy, mathematical notation, and explicit citation markers.

## 0.2. Problem statement & project pipeline 

In real usage, references inside raw LaTeX manuscripts are noisy, inconsistent, and evolving across revisions. Authors can define references manually using unstructured `\bibitem` commands, abbreviate venue names, omit publication years, or rename citation keys across versions of the same preprint. A paper can therefore cite the same work under different keys and different wording in version 1 and version 3, for example. Linking these entries to the works they actually refer to is hard unless the sources are cleaned and standardized first.

This project builds an automated, end-to-end pipeline that handles that problem. It takes the multi-version LaTeX sources of arXiv papers, together with citation metadata from Semantic Scholar, and produces a standardized, deduplicated set of tables. The tables feed a reference matching system, which is evaluated with MRR@5. The work is split into three cumulative milestones, shown below. This report covers Milestones 1 and 2.

```{mermaid}
flowchart LR
    subgraph M1["Milestone 1: Data Crawling"]
        A["arXiv API & OAI-PMH<br/>(25,000 IDs, all versions v1..vN)"] --> B["Source download &<br/>.tex/.bib extraction"]
        C["Semantic Scholar<br/>batch API"] --> D["Working corpus<br/>(papers, references)"]
        B --> D
    end
    subgraph M2["Milestone 2: Parsing & Standardization"]
        D --> E["Root detection &<br/>\input/\include resolution"]
        E --> F["Hierarchy tree, cleanup,<br/>math normalization, BibTeX"]
        F --> G["Cross-version deduplication<br/>(elements, hierarchy, bibtex)"]
    end
    subgraph M3["Milestone 3: Reference Matching"]
        G --> H["Labelling, feature<br/>engineering, ranking model"]
        H --> I["MRR@5 evaluation &<br/>Hugging Face dataset"]
    end
```

* Milestone 1(Data scraping/crawling): Resolve 25,000 assigned arXiv paper IDs, discover and download all historical revisions from the arXiv source mirror, strip heavy figure binaries in memory by extracting strictly .tex and .bib files, and extract reference metadata via the Semantic Scholar API. Output is the corpus and the papers + references tables.
* Milestone 2 (Data parsing & standardization): For each version we find the root .tex file, follow \input and \include commands to gather the files that are actually compiled, and build a document hierarchy down to sentences, block formulas and figures. We clean the LaTeX and normalize the mathematics, convert \bibitem entries into BibTeX, and deduplicate across versions, both reference entries and full-text elements. The output is the elements, hierarchy and bibtex tables.
* Milestone 3 (Reference matching **expected workflow**): We match each standardized BibTeX entry to the correct arXiv-keyed reference crawled in Milestone 1. The pipeline covers data cleaning, labelling, feature engineering, modeling and evaluation, and ends in the predictions table. All six tables will be published as one Parquet dataset on Hugging Face.

## 0.3. Repository links

In accordance with the project guidelines, the source code repository and the compiled Jupyter notebook report repository are maintained separately.

* Shared source code repo: https://github.com/kjwce/HCMUS_CSC14119_Group8_arXivProject
* Jupyter book report: https://github.com/kjwce/HCMUS_CSC14119_Group8_JupyterBook

All tables and figures in the results sections of this book are dynamically generated during book compilation from the crawler and parser logs (execute_notebooks: force in `_config.yml`).

## 0.4. Team roles & pre-assigned data scopes

Our team of five was assigned five contiguous blocks of arXiv IDs from `2603` to `2606` (March - June 2026). Each block holds exactly 5,000 papers, making it so that the corpus has 25,000 papers in total. Each member crawled their own block on a separate CPU-only Google Colab instance. Each instance made at most one request every three seconds on a single connection, per arXiv usage policy. Each member will also publish their own range as a separate Hugging Face dataset, as the course requires. The exact ID ranges are listed below. How a range becomes a concrete ID list, including the month-boundary case in Range 1, will be expanded upon later.

|  Range  | Member           | Assigned start ID | Assigned end ID |                                            M1 responsibilities                                            | M2 responsibilities            | M3 responsibilities | Notes |
| :-----: | :--------------- | :---------------: | :-------------: | :-------------------------------------------------------------------------------------------------------: | :----------------------------- | ------------------- | ----- |
| Range 1 | Nguyễn Gia Kiệt  |   `2605.27338`    |  `2606.00733`   |                                          Main scraper developer                                           | Parser development and testing | (TBD)               |       |
| Range 2 | Bùi Lê Bảo Ngọc  |   `2605.22338`    |  `2605.27337`   |               Scraper development and testing; documentation (2. Data crawling methodology)               | Parser testing; documentation  | (TBD)               |       |
| Range 3 | Tống Vân Anh     |   `2604.10535`    |  `2604.15534`   |                                            Main scraper tester                                            | Main parser tester             | (TBD)               |       |
| Range 4 | Nguyễn Tiến Hưng |   `2603.05580`    |  `2603.10579`   |                                      Scraper development and testing                                      | Main parser developer          | (TBD)               |       |
| Range 5 | Nguyễn Duy Đức   |   `2606.10734`    |  `2606.15733`   | Scraper development and testing; documentation (1. Introduction, and 3. Results & performance evaluation) | Parser testing; documentation  | (TBD)               |       |

(**Note: As of 02/11, M3 responsibilities for each member has yet to be announced; this will be agreed upon later in the duration of the project)**

Beyond the roles above, every member ran the shared scraper on their own range and collected the results.
