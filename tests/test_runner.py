from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.m2 import normalize
from src.m2.models import Node
from src.m2.run_m2 import PARSER_SCHEMA_VERSION, _normalize_tree, _write_paper_stats_jsonl, process_paper


class RunnerIntegrationTests(unittest.TestCase):
    def test_end_to_end_paper_outputs_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "raw" / "2603.00001"
            version = paper / "v1"
            version.mkdir(parents=True)
            (paper / "metadata.json").write_text(
                json.dumps(
                    {
                        "arxiv_id": "2603.00001",
                        "versions": [
                            {"version": "v1", "source_status": "ok"},
                            {"version": "v2", "source_status": "pdf_only"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (version / "main.tex").write_text(
                r"""\documentclass{article}
\begin{document}
\title{Must not leak}
\author{Hidden Author}
\begin{abstract}Abstract sentence.\end{abstract}
\section{Intro}
A cited sentence \cite{paper2020}.
\begin{align}x &= y \\ z &= q\end{align}
\begin{deluxetable*}{lc}A & B \\\end{deluxetable*}
\bibliography{refs}
\end{document}
""",
                encoding="utf-8",
            )
            (version / "refs.bib").write_text(
                "@article{paper2020, title={A Paper}, author={Doe, Jane}, "
                "year={2020}, journal={Journal}}\n",
                encoding="utf-8",
            )

            result = process_paper(str(paper), str(root / "processed"))
            resumed = process_paper(str(paper), str(root / "processed"))
            publication = root / "processed" / "2603-00001"
            stale_stats = json.loads((publication / "stats.json").read_text(encoding="utf-8"))
            stale_stats["parser_schema_version"] = PARSER_SCHEMA_VERSION - 1
            (publication / "stats.json").write_text(json.dumps(stale_stats), encoding="utf-8")
            rebuilt = process_paper(str(paper), str(root / "processed"))
            stats = json.loads((publication / "stats.json").read_text(encoding="utf-8"))
            elements = [
                json.loads(line)
                for line in (publication / "elements.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            aggregate_count = _write_paper_stats_jsonl(root / "processed", ["2603-00001"])
            aggregate = [
                json.loads(line)
                for line in (root / "processed" / "paper_stats.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "ok")
        self.assertEqual(resumed["status"], "skipped")
        self.assertEqual(rebuilt["status"], "ok")
        self.assertEqual(resumed["versions_parsed"], 1)
        self.assertEqual(resumed["elements_after_dedup"], result["elements_after_dedup"])
        self.assertEqual((result["versions_parsed"], result["empty_versions"]), (1, 1))
        self.assertEqual(stats["gather"]["v1"]["root"], "main.tex")
        self.assertEqual(stats["bibtex_source_counts_before_dedup"], {"bib": 1})
        self.assertEqual(stats["unresolved_citation_keys"], {})
        self.assertEqual(stats["parser_schema_version"], PARSER_SCHEMA_VERSION)
        self.assertEqual((aggregate_count, len(aggregate)), (1, 1))
        self.assertEqual(aggregate[0]["arxiv_id"], "2603-00001")
        self.assertTrue(any(row["kind"] == "abstract" for row in elements))
        self.assertFalse(any("Must not leak" in row["content"] or "Hidden Author" in row["content"] for row in elements))
        self.assertTrue(any(row["kind"] == "formula" and "\\begin{aligned}" in row["content"] for row in elements))
        self.assertTrue(any(row["kind"] == "figure" and "deluxetable" in row["content"] for row in elements))


    def test_sentences_empty_after_cleanup_are_dropped(self) -> None:
        root = Node("document", "Document")
        section = root.add(Node("section", "Intro"))
        section.add(Node("sentence", r"\label{sec:intro}"))
        section.add(Node("sentence", "Kept sentence."))
        section.add(Node("sentence", r"\vspace{2mm} "))
        _normalize_tree(root, normalize, {})
        self.assertEqual([child.content for child in section.children], ["Kept sentence."])


if __name__ == "__main__":
    unittest.main()
