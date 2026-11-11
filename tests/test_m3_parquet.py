import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.m3.export_parquet import (
    add_predictions, canonical, export_corpus, iter_parquet,
    validate_dataset, validate_prediction,
)
from src.m3.parquet_schema import SCHEMAS


class ParquetExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.m1, self.m2, self.output = [self.root / name for name in ("m1", "m2", "dataset")]
        self.m1.mkdir()
        self.m2.mkdir()
        self.write_json(self.m2 / "m2_stats.json", {"parser_schema_version": 3})
        for pid in ("2603.00010", "2603.00020"):
            p1, p2 = self.m1 / pid.replace(".", "-"), self.m2 / pid.replace(".", "-")
            p1.mkdir()
            p2.mkdir()
            self.write_json(p1 / "metadata.json", {
                "arxiv_id": pid, "title": "Example", "authors": ["A. Author"],
                "submission_date": "2026-03-01", "revised_dates": [], "venue": None,
                "versions": [{"version": "v1", "source_status": "ok"}],
            })
            self.write_json(p1 / "references.json", [] if pid.endswith("20") else [
                {"arxiv_id": "0804.4138", "title": "Old ID", "authors": ["B. Author"]},
                {"arxiv_id": "2406.18510", "title": "Trailing zero", "authors": ["C. Author"]},
                {"arxiv_id": None, "title": "Not on arXiv"},
            ])
            elements = [] if pid.endswith("20") else [
                {"arxiv_id": pid.replace(".", "-"), "element_id": "root1", "kind": "document", "content": "v1 root"},
                {"arxiv_id": pid.replace(".", "-"), "element_id": "root2", "kind": "document", "content": "v2 root"},
                {"arxiv_id": pid.replace(".", "-"), "element_id": "shared", "kind": "sentence", "content": "One shared sentence"},
            ]
            edges = [] if pid.endswith("20") else [
                {"arxiv_id": pid, "version": 1, "element_id": "shared", "parent_id": "root1"},
                {"arxiv_id": pid, "version": 2, "element_id": "shared", "parent_id": "root2"},
            ]
            bib = [] if pid.endswith("20") else [{
                "arxiv_id": pid, "citation_key": "test", "entry_type": "article", "title": "Trailing zero",
                "authors": ["C. Author"], "year": None, "doi": None, "journal": None, "url": None,
            }]
            for name, rows in (("elements", elements), ("hierarchy", edges), ("bibtex", bib)):
                self.write_jsonl(p2 / f"{name}.jsonl", rows)
            self.write_json(p2 / "stats.json", {
                "elements_after_dedup": len(elements), "hierarchy_edges": len(edges),
                "bibtex_entries_after_dedup": len(bib), "empty_versions": [1] if not elements else [],
            })
            (p2 / "_SUCCESS").write_text("ok", encoding="utf-8")

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def export(self):
        args = Namespace(m1=self.m1, m2=self.m2, output=self.output, limit=None, batch_rows=1, shard_rows=1)
        with contextlib.redirect_stdout(io.StringIO()):
            export_corpus(args)

    def test_full_base_export_preserves_ids_lists_null_schema_and_cross_version_edges(self):
        self.export()
        report = validate_dataset(self.output)
        self.assertEqual(report["row_counts"], {"papers": 2, "references": 2, "elements": 3, "hierarchy": 2, "bibtex": 1})
        self.assertFalse(report["ready_for_final_submission"])
        references = list(iter_parquet(self.output, "references"))
        self.assertEqual({r["ref_arxiv_id"] for r in references}, {"0804.4138", "2406.18510"})
        self.assertEqual(list(iter_parquet(self.output, "bibtex"))[0]["year"], None)
        self.assertEqual(list(iter_parquet(self.output, "papers"))[0]["authors"], ["A. Author"])
        year_type = pq.ParquetFile(next((self.output / "data" / "bibtex").glob("*.parquet"))).schema_arrow.field("year").type
        self.assertTrue(pa.types.is_int32(year_type))
        card = (self.output / "README.md").read_text(encoding="utf-8")
        self.assertIn("DRAFT", card)
        self.assertNotIn("config_name: predictions", card)

    def test_add_real_predictions_checks_frozen_labels_and_final_card(self):
        self.export()
        labels = self.root / "labels.jsonl"
        partitions = self.root / "partitions.jsonl"
        predictions = self.root / "predictions.jsonl"
        self.write_jsonl(labels, [{"arxiv_id": "2603.00010", "citation_key": "test", "groundtruth_arxiv_id": "2406.18510"}])
        self.write_jsonl(partitions, [{"arxiv_id": "2603.00010", "partition": "test"}])
        row = {"arxiv_id": "2603.00010", "partition": "test", "citation_key": "test", "groundtruth_arxiv_id": "2406.18510", "cand_1": "2406.18510", "cand_2": "0804.4138", "cand_3": None, "cand_4": None, "cand_5": None}
        self.write_jsonl(predictions, [{**row, "groundtruth_arxiv_id": None}])
        args = Namespace(dataset=self.output, labels=labels, partitions=partitions, input=predictions, batch_rows=1, shard_rows=1)
        with self.assertRaisesRegex(ValueError, "frozen labels"):
            add_predictions(args)
        self.assertFalse((self.output / "data" / "predictions").exists())
        self.write_jsonl(predictions, [row])
        with contextlib.redirect_stdout(io.StringIO()):
            add_predictions(args)
        report = validate_dataset(self.output)
        self.assertTrue(report["ready_for_final_submission"])
        self.assertEqual(report["row_counts"]["predictions"], 1)
        self.assertTrue((self.output / "_FINAL_DATASET_SUCCESS").exists())
        self.assertIn("config_name: predictions", (self.output / "README.md").read_text(encoding="utf-8"))

    def test_corrupt_hierarchy_never_marks_export_successful(self):
        p = self.m2 / "2603-00010" / "hierarchy.jsonl"
        self.write_jsonl(p, [{"arxiv_id": "2603.00010", "version": 1, "element_id": "shared", "parent_id": "missing"}])
        with self.assertRaisesRegex(ValueError, "Dangling"):
            self.export()
        self.assertFalse((self.output / "_BASE_TABLES_SUCCESS").exists())

    def test_prediction_rejects_float_ids_truncated_ids_and_candidates_of_other_papers(self):
        for value in (2406.18510, "2406.1851"):
            with self.assertRaises(ValueError):
                if isinstance(value, float):
                    canonical(value)
                else:
                    row = {"arxiv_id": "2603.00010", "partition": "test", "citation_key": "test", "groundtruth_arxiv_id": value, **{f"cand_{i}": None for i in range(1, 6)}}
                    validate_prediction(row, {"2406.18510"})
        self.assertTrue(pa.types.is_dictionary(SCHEMAS["predictions"].field("partition").type))


if __name__ == "__main__":
    unittest.main()
