import unittest
from unittest.mock import patch

from src.m3.audit import audit_manual_shortlist
from src.m3.models import BibEntry, Reference


def _entry(paper, number, complete=True):
    return BibEntry(
        paper,
        f"key{number}",
        "article",
        f"Shared title {number}",
        ("Alice Smith",) if complete else (),
        2020 if complete else None,
        None,
        None,
        None,
    )


class AuditTests(unittest.TestCase):
    @patch("src.m3.audit.load_joined_paper")
    def test_recommends_four_easy_and_one_medium(self, load):
        def paper_data(arxiv_id, *_):
            number = int(arxiv_id[-1])
            entries = [_entry(arxiv_id, index) for index in range(25)]
            exact = 22 if number < 5 else 12
            refs = [
                Reference(arxiv_id, f"2001.{index:05d}", f"Shared title {index}", (), 2020, None, None)
                for index in range(exact)
            ]
            return entries, refs

        load.side_effect = paper_data
        rows = [
            {
                "range": "r1",
                "arxiv_id": f"2603.0000{number}",
                "bibtex_entries": 25,
                "arxiv_references": 25,
                "shortlist_rank": number + 1,
                "recommended": number < 5,
            }
            for number in range(10)
        ]
        audited = audit_manual_shortlist(rows, {}, {})
        chosen = [row for row in audited if row["final_recommended"]]
        self.assertEqual(len(chosen), 5)
        self.assertEqual(sum(row["difficulty"] == "medium" for row in chosen), 1)
        self.assertTrue(all(row["final_rank"] for row in chosen))


if __name__ == "__main__":
    unittest.main()
