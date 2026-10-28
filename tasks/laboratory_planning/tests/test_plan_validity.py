import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lab_inference.features import structure
from lab_inference.plan_schema import is_valid_sequence, parse_steps


class PlanValidityTests(unittest.TestCase):
    def validity(self, code, pool=None):
        return is_valid_sequence(
            parse_steps(code), raw_inputs={'sample'},
            action_pool_names={'transfer'} if pool is None else pool,
        )

    def test_undefined_names_inside_data_expressions(self):
        expressions = [
            'missing', '[missing]', "{'sample': missing}", 'missing + 1',
            '[sample, (missing,)]', 'sample[missing]', 'missing.volume',
            "{missing: sample}", 'sample if missing else sample', "f'{missing}'",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                valid, details = self.validity(f'transfer(material={expression})')
                self.assertFalse(valid)
                self.assertIn('missing', details['unknown_vars'])

    def test_literals_and_defined_nested_inputs(self):
        expressions = [
            'sample', '[sample]', "{'sample': sample}", 'sample + 1',
            "['missing', -1, True]", '(sample, 2)', 'sample[0]', 'sample.volume',
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertTrue(self.validity(f'transfer(material={expression})')[0])

    def test_nested_forward_reference_is_rejected(self):
        valid, details = self.validity(
            'transfer(material=[prepared])\nprepared = transfer(material=sample)'
        )
        self.assertFalse(valid)
        self.assertEqual(details['future_generated_vars'], ['prepared'])

    def test_earlier_output_and_reused_handle_are_valid(self):
        self.assertTrue(self.validity(
            'prepared = transfer(material=sample)\n'
            "prepared = transfer(material={'sample': [prepared]})"
        )[0])

    def test_feature_legality_uses_recursive_variable_check(self):
        task = {
            'id': 'nested-input', 'context': 'Transfer a sample', 'goal': 'transfer',
            'constraints': [], 'available_inputs': json.dumps([{'name': 'sample'}]),
            'action_pool': 'def transfer(material):\n    pass',
        }
        self.assertFalse(structure(task, 'transfer(material=[missing])')[1])
        self.assertTrue(structure(task, "transfer(material={'sample': [sample]})")[1])


if __name__ == '__main__':
    unittest.main()
