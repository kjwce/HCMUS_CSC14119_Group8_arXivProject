from __future__ import annotations

import unittest

import pandas as pd

from src.m3.feature_evidence import ARXIV_IN_TEXT, feature_stats, mrr_ceiling, unique_top1


class FeatureEvidenceTests(unittest.TestCase):
    def test_arxiv_ids_found_in_bibtex_text(self) -> None:
        text = "https://arxiv.org/abs/2501.12948 doi:10.48550/arXiv.2405.11143 arXiv:hep-th/9901001"
        self.assertEqual(
            [m.group(1).lower() for m in ARXIV_IN_TEXT.finditer(text)],
            ["2501.12948", "2405.11143", "hep-th/9901001"],
        )

    def test_unique_top1_counts_only_strict_winners(self) -> None:
        pairs = pd.DataFrame({
            "arxiv_id": ["p"] * 4,
            "citation_key": ["a", "a", "b", "b"],
            "gt": ["r1", "r1", "r3", "r3"],
            "label": [1, 0, 1, 0],
            "score": [0.9, 0.2, 0.5, 0.5],  # entry b is a tie, so not a unique win
        })
        self.assertEqual(unique_top1(pairs, "score"), {"share": 0.5, "entries": 2})

    def test_feature_stats_auc_and_medians(self) -> None:
        pairs = pd.DataFrame({"label": [1, 1, 0, 0], "f": [0.9, 0.8, 0.1, 0.2]})
        self.assertEqual(feature_stats(pairs, ["f"])["f"],
                         {"match_median": 0.85, "nonmatch_median": 0.15, "auc": 1.0})

    def test_mrr_ceiling_is_share_with_ground_truth(self) -> None:
        pred = pd.DataFrame({
            "partition": ["test"] * 4,
            "groundtruth_arxiv_id": ["x", None, "y", None],
            "is_manual": [True, True, False, False],
        })
        self.assertEqual(mrr_ceiling(pred)["test"],
                         {"entries": 4, "ceiling": 0.5, "manual": 0.5, "automatic": 0.5})


if __name__ == "__main__":
    unittest.main()
