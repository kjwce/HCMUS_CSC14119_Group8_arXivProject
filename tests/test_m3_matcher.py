import unittest

from src.m3.matcher import automatic_label, label_publication, rank_candidates
from src.m3.models import BibEntry, Reference


def bib(key, title, authors=("Alice Smith",), year=2020, url=None):
    return BibEntry("2603.05580", key, "article", title, authors, year, None, None, url)


def ref(arxiv_id, title, authors=("A. Smith",), year=2020):
    return Reference("2603.05580", arxiv_id, title, authors, year, None, None)


class MatcherTests(unittest.TestCase):
    def test_matching_ranks_the_same_work_first(self):
        entry = bib("key", "A robust method for reference matching")
        references = [
            ref("2001.00001", "Unrelated quantum result", ("B. Jones",), 2017),
            ref("2002.00002", "A robust method for reference matching"),
        ]
        ranked = rank_candidates(entry, references)
        label = automatic_label(entry, references)
        self.assertEqual(ranked[0].ref_arxiv_id, "2002.00002")
        self.assertEqual(label["groundtruth_arxiv_id"], "2002.00002")
        self.assertTrue(label["accepted"])

    def test_embedded_arxiv_id_is_an_exact_rule(self):
        entry = bib("key", "Noisy title", url="https://arxiv.org/abs/2002.00002v3")
        label = automatic_label(entry, [ref("2002.00002", "Different metadata")])
        self.assertEqual(label["groundtruth_arxiv_id"], "2002.00002")
        self.assertEqual(label["label_method"], "embedded_arxiv_id")

    def test_every_entry_gets_a_row_including_no_match(self):
        entries = [bib("one", "No match one"), bib("two", "No match two")]
        rows = label_publication(entries, [])
        self.assertEqual([row["citation_key"] for row in rows], ["one", "two"])
        self.assertTrue(all(row["groundtruth_arxiv_id"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
