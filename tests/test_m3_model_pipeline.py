import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pyarrow.parquet as pq

from src.m3.evaluate import evaluate_predictions, reciprocal_rank
from src.m3.feature_engineering import prepare_features
from src.m3.model_data import handoff_labels, merge_labels, read_jsonl
from src.m3.train_model import train
from src.m3.prediction import rank_candidates


class ModellingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.handoff = self.root / "handoff"
        self.handoff.mkdir()
        self.features = self.root / "features"
        self.parts = []
        self.bib = []
        self.refs = []
        self.labels = []
        for i, split in enumerate(("train", "train", "valid", "test"), 1):
            pid = f"2603.{i:05d}"
            source = "manual" if i % 2 else "automatic"
            self.parts.append({"arxiv_id": pid, "partition": split, "label_source": source})
            title = "Alpha matching training" if split == "train" else "zzqzzqzzq heldout title"
            for key, truth, bib_title in (("a", "2406.18510", title), ("b", None, "Unrelated unknown gamma")):
                self.bib.append({"arxiv_id": pid, "paper_arxiv_id": pid, "citation_key": key, "title": bib_title, "authors": ["A. Author"], "year": 2024, "doi": None, "journal": None, "url": None})
                self.labels.append({"arxiv_id": pid, "citation_key": key, "groundtruth_arxiv_id": truth, "partition": split, "label_source": source})
            for rid, ref_title in (("2406.18510", title), ("0804.4138", "Other beta work")):
                self.refs.append({"arxiv_id": rid, "ref_arxiv_id": rid, "arxiv_key": rid, "paper_arxiv_id": pid, "title": ref_title, "authors": ["A. Author"], "year": 2024})
        for name, rows in (("bibtex", self.bib), ("references", self.refs), ("labels", self.labels), ("partitions", self.parts)):
            self.write(name, rows)

    def write(self, name, rows):
        (self.handoff / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_merge_normalizes_hyphens_preserves_zeroes_and_null_labels(self):
        pairs = pd.DataFrame([
            {"arxiv_id": "2603-00001", "citation_key": "a", "ref_arxiv_id": "2406-18510"},
            {"arxiv_id": "2603-00001", "citation_key": "b", "ref_arxiv_id": "0804.4138"},
        ])
        labels, _ = handoff_labels(self.handoff)
        merged = merge_labels(pairs, labels)
        self.assertEqual(merged["label"].tolist(), [1, 0])
        self.assertEqual(merged["ref_arxiv_id"].tolist(), ["2406.18510", "0804.4138"])
        duplicate = pd.concat([labels, labels.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge_labels(pairs, duplicate)

    def test_mrr_counts_no_match_as_zero_and_rejects_missing_predictions(self):
        self.assertEqual(reciprocal_rank("0804.4138", ["2406.18510", "0804.4138"]), .5)
        self.assertEqual(reciprocal_rank(None, [None] * 5), 0)
        labels, _ = handoff_labels(self.handoff)
        rows = []
        for row in self.labels:
            rows.append({"arxiv_id": row["arxiv_id"], "citation_key": row["citation_key"], "groundtruth_arxiv_id": row["groundtruth_arxiv_id"], "partition": row["partition"], "cand_1": "2406.18510", **{f"cand_{i}": None for i in range(2, 6)}})
        result = evaluate_predictions(rows, labels)
        self.assertEqual(result["overall"]["mrr_at_5"], .5)
        self.assertEqual(result["overall"]["mrr_at_5_matched_only"], 1)
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_predictions(rows[:-1], labels)

    def test_top_five_combines_entry_chunks_and_breaks_ties_by_string_id(self):
        labels, _ = handoff_labels(self.handoff)
        labels = labels.iloc[:1]
        first = pd.DataFrame([
            {"arxiv_id": "2603.00001", "citation_key": "a", "ref_arxiv_id": "2406.18510", "combo_score": .9},
            {"arxiv_id": "2603.00001", "citation_key": "a", "ref_arxiv_id": "0804.4138", "combo_score": .9},
        ])
        second = pd.DataFrame([
            {"arxiv_id": "2603.00001", "citation_key": "a", "ref_arxiv_id": "1901.00010", "combo_score": .95},
        ])
        with patch("src.m3.prediction.iter_feature_batches", return_value=iter([first, second])):
            result = rank_candidates(None, labels, [])
        self.assertEqual([result[0][f"cand_{i}"] for i in range(1, 4)], ["1901.00010", "0804.4138", "2406.18510"])

    def test_end_to_end_train_only_vocabulary_all_candidates_and_final_predictions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = prepare_features(self.handoff, self.features, block=1, shard_rows=3)
        self.assertEqual(manifest["pair_rows"], 16)
        self.assertEqual(len(manifest["feature_columns"]), 43)
        vectorizer = joblib.load(self.features / "tfidf.joblib")
        self.assertNotIn("zzq", vectorizer.vocabulary_)
        self.assertIsNone(manifest["max_cands"])
        output = self.root / "models"
        with contextlib.redirect_stdout(io.StringIO()):
            train(Namespace(features=self.features, handoff=self.handoff, output=output, epochs=1, negative_cap=10, seed=7, tree_iterations=2))
        predictions = read_jsonl(output / "predictions.jsonl")
        self.assertEqual(len(predictions), 8)
        self.assertEqual(set(row["cand_1"] for row in predictions) | set(row["cand_2"] for row in predictions), {"0804.4138", "2406.18510"})
        self.assertTrue(all(row["cand_3"] is None for row in predictions))
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        self.assertFalse(selection["test_used_for_selection"])
        self.assertIn("groundtruth_arxiv_id", pq.ParquetFile(output / "predictions.parquet").schema_arrow.names)
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["overall"]["no_match_entries"], 4)
        self.assertTrue((output / "plots" / "selected_model_mrr.png").exists())


if __name__ == "__main__":
    unittest.main()
