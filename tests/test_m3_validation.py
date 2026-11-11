import unittest
from unittest.mock import patch

from src.m3.models import BibEntry, Reference
from src.m3.validation import validate_automatic_labels


class AutomaticLabelValidationTests(unittest.TestCase):
    @patch("src.m3.validation.load_joined_paper")
    def test_complete_publication_scoped_labels_pass(self, load):
        load.return_value = (
            [
                BibEntry("2603.00001", "a", "article", "A", (), 2020, None, None, None),
                BibEntry("2603.00001", "b", "article", "B", (), 2020, None, None, None),
            ],
            [Reference("2603.00001", "2001.00001", "A", (), 2020, None, None)],
        )
        pool = [{"arxiv_id": "2603.00001", "bibtex_entries": 2}]
        labels = [
            {
                "arxiv_id": "2603.00001",
                "citation_key": "a",
                "groundtruth_arxiv_id": "2001.00001",
                "label_source": "automatic",
                "accepted": True,
            },
            {
                "arxiv_id": "2603.00001",
                "citation_key": "b",
                "groundtruth_arxiv_id": None,
                "label_source": "automatic",
                "accepted": False,
            },
        ]
        summary = validate_automatic_labels(pool, labels, {}, {})
        self.assertEqual(summary["automatic_label_rows"], 2)
        self.assertEqual(summary["labels_accepted"], 1)

    @patch("src.m3.validation.load_joined_paper")
    def test_groundtruth_outside_publication_is_rejected(self, load):
        load.return_value = (
            [BibEntry("2603.00001", "a", "article", "A", (), 2020, None, None, None)],
            [],
        )
        with self.assertRaisesRegex(ValueError, "outside publication candidates"):
            validate_automatic_labels(
                [{"arxiv_id": "2603.00001", "bibtex_entries": 1}],
                [
                    {
                        "arxiv_id": "2603.00001",
                        "citation_key": "a",
                        "groundtruth_arxiv_id": "2001.00001",
                        "label_source": "automatic",
                        "accepted": True,
                    }
                ],
                {},
                {},
            )


if __name__ == "__main__":
    unittest.main()
