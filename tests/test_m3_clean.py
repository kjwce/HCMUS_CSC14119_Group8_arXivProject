import unittest

from src.m3.clean import (
    author_surnames,
    normalize_arxiv_id,
    normalize_doi,
    normalize_text,
    title_tokens,
)


class CleaningTests(unittest.TestCase):
    def test_arxiv_identifiers_are_unversioned_and_canonical(self):
        self.assertEqual(normalize_arxiv_id("arXiv:2603-05580v2"), "2603.05580")
        self.assertEqual(
            normalize_arxiv_id("https://arxiv.org/pdf/hep-th/9901001v3.pdf"),
            "hep-th/9901001",
        )
        self.assertIsNone(normalize_arxiv_id("Results from 2020-2021"))

    def test_text_and_identifier_cleaning(self):
        self.assertEqual(normalize_text(r"\textbf{Café} \& Models"), "cafe and models")
        self.assertEqual(normalize_doi("https://doi.org/10.1000/ABC."), "10.1000/abc")
        self.assertEqual(title_tokens("The Model of a System"), {"model", "system"})

    def test_author_surnames(self):
        self.assertEqual(
            author_surnames(["F. Colombo", "Ludwig van Beethoven"]),
            {"colombo", "van beethoven"},
        )


if __name__ == "__main__":
    unittest.main()
