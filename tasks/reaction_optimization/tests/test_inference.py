"""Check decision semantics, malformed measurements and portable CLI behavior."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suzuki_inference import expected_improvement, load_prior, recommend


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = load_prior()
        cls.example = json.loads((ROOT / "examples/observations.json").read_text())
        cls.document = json.loads((ROOT / "assets/suzuki_prior.json").read_text())

    def test_ei_at_incumbent_has_closed_form_value(self):
        score = expected_improvement(np.array([50.0]), np.array([6.0]), 50.0, 0.0)
        self.assertAlmostEqual(float(score[0]), 6.0 / math.sqrt(2 * math.pi), places=13)

    def test_measurements_are_excluded_and_order_does_not_change_decision(self):
        first = recommend(self.example, self.prior)
        self.assertNotIn(first["candidate_index"], [r["candidate_index"] for r in self.example])
        self.assertEqual(first, recommend(list(reversed(self.example)), self.prior))
        extended = self.example + [{"candidate_index": first["candidate_index"], "yield": 75.0}]
        self.assertNotEqual(first["candidate_index"], recommend(extended, self.prior)["candidate_index"])

    def test_ties_choose_smallest_unobserved_id(self):
        document = copy.deepcopy(self.document)
        document["candidates"] = document["candidates"][:3]
        for candidate in document["candidates"]:
            candidate["predicted_yield_percent"] = 50.0
            candidate["calibrated_std_percent_points"] = 6.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            path.write_text(json.dumps(document))
            self.assertEqual(recommend([{"candidate_index": 0, "yield": 50.0}], path)["candidate_index"], 1)

    def test_invalid_measurements_raise_clear_errors(self):
        bad_inputs = [None, {}, [], [None], [{}]]
        for candidate_id in [True, 1.5, "1", -1, self.prior.candidate_count]:
            bad_inputs.append([{"candidate_index": candidate_id, "yield": 50.0}])
        for value in [True, "50", float("nan"), float("inf"), -1, 101]:
            bad_inputs.append([{"candidate_index": 0, "yield": value}])
        bad_inputs.append([{"candidate_index": 0, "yield": 1.0}, {"candidate_index": 0, "yield": 2.0}])
        for observations in bad_inputs:
            with self.subTest(observations=observations), self.assertRaises(ValueError):
                recommend(observations, self.prior)

    def test_complete_pool_has_no_recommendation(self):
        observations = [{"candidate_index": i, "yield": 50.0} for i in range(self.prior.candidate_count)]
        with self.assertRaisesRegex(ValueError, "All candidate"):
            recommend(observations, self.prior)

    def test_malformed_priors_are_rejected(self):
        changes = [
            ("candidate_index", 1.5),
            ("candidate_index", 1),
            ("original_table_index", -1),
            ("predicted_yield_percent", float("nan")),
            ("predicted_yield_percent", 101),
            ("calibrated_std_percent_points", 0),
            ("conditions", {}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            for key, value in changes:
                document = copy.deepcopy(self.document)
                document["candidates"][0][key] = value
                path.write_text(json.dumps(document))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_prior(path)
            document = copy.deepcopy(self.document)
            document["candidates"][1]["original_table_index"] = document["candidates"][0]["original_table_index"]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_prior(path)

    def test_cli_runs_from_another_directory_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "recommendation.json"
            command = [sys.executable, str(ROOT / "infer.py"), "--observations", str(ROOT / "examples/observations.json"), "--output", str(destination)]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(destination.read_text()), recommend(self.example, self.prior))
            original = destination.read_bytes()
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(destination.read_bytes(), original)

    def test_cli_reports_invalid_input_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.json"
            path.write_text('[{"candidate_index": 1.5, "yield": 40}]')
            result = subprocess.run([sys.executable, str(ROOT / "infer.py"), "--observations", str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be an integer", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_example_recommendation_matches_released_code(self):
        expected = json.loads((ROOT / "examples/recommendation.json").read_text())
        self.assertEqual(recommend(self.example, self.prior), expected)


if __name__ == "__main__":
    unittest.main()
