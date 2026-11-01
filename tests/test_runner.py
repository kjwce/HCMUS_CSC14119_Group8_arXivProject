from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.m2.run_m2 import process_paper


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
\section{Intro}
A cited sentence \cite{paper2020}.
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
            stats = json.loads((publication / "stats.json").read_text(encoding="utf-8"))
            elements = [
                json.loads(line)
                for line in (publication / "elements.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "ok")
        self.assertEqual(resumed["status"], "skipped")
        self.assertEqual(resumed["versions_parsed"], 1)
        self.assertEqual(resumed["elements_after_dedup"], result["elements_after_dedup"])
        self.assertEqual((result["versions_parsed"], result["empty_versions"]), (1, 1))
        self.assertEqual(stats["gather"]["v1"]["root"], "main.tex")
        self.assertEqual(stats["bibtex_source_counts_before_dedup"], {"bib": 1})
        self.assertEqual(stats["unresolved_citation_keys"], {})
        self.assertTrue(any(row["kind"] == "figure" and "deluxetable" in row["content"] for row in elements))


if __name__ == "__main__":
    unittest.main()
