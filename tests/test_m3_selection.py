import unittest

from src.m3.selection import finalize_manual_pool, select_manual_shortlist


class ManualSelectionTests(unittest.TestCase):
    def test_balanced_reproducible_shortlist(self):
        inventory = {}
        manifests = {"r1": [], "r2": []}
        for range_number in (1, 2):
            for paper_number in range(8):
                arxiv_id = f"260{range_number}.{paper_number:05d}"
                inventory[arxiv_id] = {
                    "bibtex_entries": 20 + paper_number,
                    "arxiv_references": 10,
                    "eligible_for_manual_labelling": True,
                }
                manifests[f"r{range_number}"].append(
                    {"arxiv_id": arxiv_id, "title": f"Paper {paper_number}"}
                )
        first = select_manual_shortlist(inventory, manifests, per_range=6, seed=7)
        second = select_manual_shortlist(inventory, manifests, per_range=6, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(sum(bool(row["recommended"]) for row in first), 10)
        self.assertEqual({row["range"] for row in first}, {"r1", "r2"})

    def test_filters_unmanageable_publications(self):
        inventory = {
            "2603.00001": {
                "bibtex_entries": 200,
                "arxiv_references": 10,
                "eligible_for_manual_labelling": True,
            }
        }
        with self.assertRaisesRegex(ValueError, "only 0 candidates"):
            select_manual_shortlist(
                inventory,
                {"r1": [{"arxiv_id": "2603.00001"}]},
                per_range=1,
            )

    def test_final_pool_requires_five_unique_rows_per_range(self):
        rows = []
        for range_name in ("r1", "r2"):
            for number in range(5):
                rows.append(
                    {
                        "range": range_name,
                        "arxiv_id": f"260{range_name[-1]}.{number:05d}",
                        "bibtex_entries": 25,
                        "arxiv_references": 15,
                        "difficulty": "easy",
                        "title": "Title",
                        "final_recommended": True,
                        "final_rank": number + 1,
                    }
                )
        final = finalize_manual_pool(rows)
        self.assertEqual(len(final), 10)
        self.assertEqual({row["range"] for row in final}, {"r1", "r2"})

    def test_final_pool_rejects_incomplete_range(self):
        with self.assertRaisesRegex(ValueError, "Each range must contain 5"):
            finalize_manual_pool(
                [
                    {
                        "range": "r1",
                        "arxiv_id": "2603.00001",
                        "final_recommended": True,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
