import json
import tempfile
import unittest
from pathlib import Path

from src.m3.data import index_corpora, inventory_rows, load_joined_paper, read_m2_schema_version


class DataReaderTests(unittest.TestCase):
    def test_join_hyphenated_m2_with_dotted_m1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m1_paper = root / "m1" / "raw" / "2603-05580"
            m2_paper = root / "m2" / "2603-05580"
            m1_paper.mkdir(parents=True)
            m2_paper.mkdir(parents=True)
            (m1_paper / "references.json").write_text(
                json.dumps(
                    [
                        {
                            "title": "Target title",
                            "authors": ["A. Author"],
                            "year": 2020,
                            "arxiv_id": "2001.00001v2",
                            "semantic_scholar_id": "s2",
                        },
                        {"title": "Not on arXiv", "arxiv_id": None},
                    ]
                ),
                encoding="utf-8",
            )
            (m2_paper / "bibtex.jsonl").write_text(
                json.dumps(
                    {
                        "arxiv_id": "2603-05580",
                        "citation_key": "key",
                        "entry_type": "article",
                        "title": "Target title",
                        "authors": ["Alice Author"],
                        "year": 2020,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "m2" / "m2_stats.json").write_text(
                json.dumps({"parser_schema_version": 3}), encoding="utf-8"
            )

            m1_index, m2_index = index_corpora(root / "m1", root / "m2")
            entries, references = load_joined_paper("2603.05580", m1_index, m2_index)
            inventory = list(inventory_rows(root / "m1", root / "m2"))

            self.assertEqual(entries[0].arxiv_id, "2603.05580")
            self.assertEqual(references[0].ref_arxiv_id, "2001.00001")
            self.assertEqual(len(references), 1)
            self.assertEqual(inventory[0].bibtex_entries, 1)
            self.assertEqual(inventory[0].arxiv_references, 1)
            self.assertEqual(read_m2_schema_version(root / "m2"), 3)


if __name__ == "__main__":
    unittest.main()
