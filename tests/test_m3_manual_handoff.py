import unittest

from src.m3.manual_handoff import calibrate, prediction, validate_label_row


class ManualHandoffTests(unittest.TestCase):
    def test_rejects_excel_truncated_id_and_no_match_with_id(self):
        with self.assertRaises(ValueError):
            validate_label_row({"match_status": "matched", "groundtruth_arxiv_id": "2406.1851"}, {"2406.18510"})
        with self.assertRaises(ValueError):
            validate_label_row({"match_status": "no_match", "groundtruth_arxiv_id": "2406.18510"}, {"2406.18510"})
        self.assertEqual(validate_label_row({"match_status": "matched", "groundtruth_arxiv_id": "0804.4138"}, {"0804.4138"}), "0804.4138")

    def test_embedded_id_rule_and_missing_candidate(self):
        row = {"best_candidate_features": {"ref_arxiv_id": "2406.18510"}, "label_method": "embedded_arxiv_id"}
        self.assertEqual(prediction(row, .90, .20), "2406.18510")
        self.assertIsNone(prediction({"best_candidate_features": None}, .68, .05))

    def test_training_calibration_rejects_false_match_and_recovers_true_match(self):
        def sample(score, margin, truth):
            return {"best_candidate_features": {"ref_arxiv_id": "2406.18510"}, "label_method": "weighted_string_similarity", "confidence": score, "margin": margin, "reviewed_groundtruth": truth}
        train = [sample(.95, .5, "2406.18510"), sample(.62, .3, "2406.18510"), sample(.60, .01, None)]
        selected = calibrate(train)["selected"]
        self.assertEqual(selected["correct_matches"], 2)
        self.assertEqual(selected["exact_correct"], 3)
        self.assertEqual(selected["match_precision"], 1.0)
        self.assertIsNone(prediction(train[-1], selected["threshold"], selected["minimum_margin"]))


if __name__ == "__main__":
    unittest.main()
