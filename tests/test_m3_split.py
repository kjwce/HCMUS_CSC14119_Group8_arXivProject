import unittest

from src.m3.split import split_publications


class SplitTests(unittest.TestCase):
    def test_five_member_split_is_by_publication_and_deterministic(self):
        manual = {f"2603.{number:05d}" for number in range(25)}
        automatic = {f"2604.{number:05d}" for number in range(30)}
        first = split_publications(manual, automatic, seed=7)
        second = split_publications(manual, automatic, seed=7)
        self.assertEqual(first, second)

        counts = {}
        for row in first:
            key = (row["label_source"], row["partition"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts[("manual", "test")], 5)
        self.assertEqual(counts[("manual", "valid")], 5)
        self.assertEqual(counts[("manual", "train")], 15)
        self.assertEqual(counts[("automatic", "test")], 5)
        self.assertEqual(counts[("automatic", "valid")], 5)
        self.assertEqual(counts[("automatic", "train")], 20)

    def test_overlapping_pools_are_rejected(self):
        with self.assertRaises(ValueError):
            split_publications(
                {"2603.00001"} | {f"2603.{i:05d}" for i in range(2, 26)},
                {"2603.00001"} | {f"2604.{i:05d}" for i in range(10)},
            )

    def test_five_member_split_requires_all_twenty_five_manual_papers(self):
        manual = {f"2603.{number:05d}" for number in range(24)}
        automatic = {f"2604.{number:05d}" for number in range(10)}
        with self.assertRaisesRegex(ValueError, "at least 25 manual"):
            split_publications(manual, automatic)


if __name__ == "__main__":
    unittest.main()
