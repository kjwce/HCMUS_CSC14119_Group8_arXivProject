from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.m2.models import Node
from src.m2.store import PublicationStore


class StoreTests(unittest.TestCase):
    @staticmethod
    def tree(heading: str, sentence: str) -> Node:
        root = Node("document", "Document")
        section = root.add(Node("section", heading))
        section.add(Node("sentence", sentence))
        return root

    def test_content_deduplicates_across_versions_but_roots_do_not(self) -> None:
        store = PublicationStore("2603.12345")
        store.add_version(1, self.tree("Introduction", "Shared sentence."))
        store.add_version(2, self.tree("Background", "Shared sentence."))

        roots = [row for row in store.elements.values() if row.kind == "document"]
        sentences = [row for row in store.elements.values() if row.kind == "sentence"]
        self.assertEqual(len(roots), 2)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(len(store.hierarchy), 4)

    def test_repeated_content_in_one_version_gets_distinct_ids(self) -> None:
        root = Node("document", "Document")
        root.add(Node("sentence", "Repeated."))
        root.add(Node("sentence", "Repeated."))
        store = PublicationStore("2603-12345")
        store.add_version(1, root)
        sentence_ids = [row.element_id for row in store.elements.values() if row.kind == "sentence"]
        self.assertEqual(len(sentence_ids), 2)
        self.assertEqual(len(set(sentence_ids)), 2)

    def test_write_creates_three_tables_and_success_marker(self) -> None:
        store = PublicationStore("2603.12345")
        store.add_version(1, self.tree("Introduction", "Hello."))
        with tempfile.TemporaryDirectory() as temporary:
            output = store.write(Path(temporary), [{"citation_key": "ref1"}])
            self.assertTrue((output / "elements.jsonl").is_file())
            self.assertTrue((output / "hierarchy.jsonl").is_file())
            self.assertTrue((output / "bibtex.jsonl").is_file())
            self.assertTrue((output / "_SUCCESS").is_file())
            stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["versions_parsed"], 1)


if __name__ == "__main__":
    unittest.main()

