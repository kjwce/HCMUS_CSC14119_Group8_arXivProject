import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from src.m3.export_parquet import iter_parquet, validate_dataset
from src.m3.parquet_schema import SCHEMAS, describe_schemas
from src.m3.split_dataset import digest, load_assignments, split_dataset, verify_projection


class RangeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "split"
        self.ids = [f"2603.{number:04d}0" for number in range(1, 6)]
        self.specs = []
        rows = {name: [] for name in SCHEMAS}
        for number, pid in enumerate(self.ids, 1):
            manifest = self.root / f"r{number}.jsonl"
            manifest.write_text(json.dumps({"arxiv_id": pid.replace(".", "-")}) + "\n", encoding="utf-8")
            self.specs.append(f"r{number}={manifest}")
            # The first paper cites a work assigned to range 5. Keep that row in r1.
            ref = self.ids[-1] if number == 1 else "2006.09661"
            rows["papers"].append({"arxiv_id": pid, "title": f"Paper {number}", "authors": ["A. Author"],
                                   "submission_date": "2026-03-01", "revised_dates": [], "venue": None})
            rows["references"].append({"arxiv_id": pid, "ref_arxiv_id": ref, "ref_title": "Reference",
                                       "ref_authors": ["B. Author"], "ref_submission_date": "2020-06-01", "ref_s2_id": None})
            if number < 5:
                rows["elements"] += [{"arxiv_id": pid, "element_id": pid + "root", "kind": "document", "content": ""},
                                      {"arxiv_id": pid, "element_id": pid + "child", "kind": "sentence", "content": "Exact text: \\alpha, café."}]
                rows["hierarchy"].append({"arxiv_id": pid, "version": 1, "element_id": pid + "child", "parent_id": pid + "root"})
            rows["bibtex"].append({"arxiv_id": pid, "citation_key": "citation", "entry_type": "article",
                                   "title": "Reference", "authors": ["B. Author"], "year": 2020,
                                   "doi": None, "journal": None, "url": None})
            rows["predictions"].append({"arxiv_id": pid, "partition": ("train", "valid", "test")[number % 3],
                                        "citation_key": "citation", "groundtruth_arxiv_id": ref if number % 2 else None,
                                        "cand_1": ref, "cand_2": None, "cand_3": None, "cand_4": None, "cand_5": None})
        for name, schema in SCHEMAS.items():
            directory = self.source / "data" / name
            directory.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows[name], schema=schema), directory / "part-00000.parquet", row_group_size=3)
        report = validate_dataset(self.source)
        (self.source / "export_report.json").write_text(json.dumps(report), encoding="utf-8")
        (self.source / "README.md").write_text("Source corpus\n", encoding="utf-8")
        (self.source / "schema.json").write_text(json.dumps(describe_schemas()), encoding="utf-8")
        self.args = Namespace(dataset=self.source, output=self.output, range_manifest=self.specs,
                              expected_per_range=1, shard_rows=1, source_repo="group/source")

    def split(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return split_dataset(self.args)

    def test_exact_projection_foreign_range_references_empty_tables_and_archives(self):
        original = {p.relative_to(self.source).as_posix(): digest(p) for p in self.source.rglob("*") if p.is_file()}
        summary = self.split()
        self.assertEqual(summary["total_publications"], 5)
        self.assertTrue((self.output / "_SPLIT_SUCCESS").is_file())
        for number, pid in enumerate(self.ids, 1):
            path = self.output / f"range_{number}_dataset"
            for table in SCHEMAS:
                expected = [row for row in iter_parquet(self.source, table) if row["arxiv_id"] == pid]
                self.assertEqual(list(iter_parquet(path, table)), expected)
            self.assertEqual(validate_dataset(path)["row_counts"]["papers"], 1)
            card = (path / "README.md").read_text(encoding="utf-8")
            self.assertEqual(card.count("- config_name:"), 6)
            self.assertIn("predictions.partition", card)
            with ZipFile(self.output / "packages" / f"range_{number}_dataset.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn(f"range_{number}_dataset/README.md", archive.namelist())
                self.assertIn("HUONG_DAN_UPLOAD.md", archive.namelist())
        self.assertEqual(list(iter_parquet(self.output / "range_1_dataset", "references"))[0]["ref_arxiv_id"], self.ids[-1])
        empty = next((self.output / "range_5_dataset/data/elements").glob("*.parquet"))
        self.assertEqual(pq.ParquetFile(empty).metadata.num_rows, 0)
        self.assertEqual(pq.ParquetFile(empty).schema_arrow, SCHEMAS["elements"])
        self.assertEqual(original, {p.relative_to(self.source).as_posix(): digest(p) for p in self.source.rglob("*") if p.is_file()})

    def test_overlapping_manifests_and_missing_source_membership_are_rejected(self):
        (self.root / "r2.jsonl").write_text(json.dumps({"arxiv_id": self.ids[0]}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            load_assignments(self.specs, 1)
        (self.root / "r2.jsonl").write_text(json.dumps({"arxiv_id": "2603.99990"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            self.split()
        self.assertFalse(self.output.exists())

    def test_projection_check_detects_changed_content_and_existing_output_is_protected(self):
        self.split()
        with self.assertRaisesRegex(ValueError, "new output"):
            self.split()
        path = self.output / "range_1_dataset/data/elements/part-00000.parquet"
        table = pq.ParquetFile(path).read()
        rows = table.to_pylist()
        rows[0]["content"] = "Changed, but count and schema still match"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS["elements"]), path)
        assignments, _, _ = load_assignments(self.specs, 1)
        outputs = {f"r{i}": self.output / f"range_{i}_dataset" for i in range(1, 6)}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "changed source row"):
            verify_projection(self.source, outputs, assignments)


if __name__ == "__main__":
    unittest.main()
