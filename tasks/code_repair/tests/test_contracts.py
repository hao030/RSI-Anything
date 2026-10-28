import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation import choose, state_key
from fixtures import execution_from_dict
from io_utils import lines


class DecisionContracts(unittest.TestCase):
    def test_current_program_wins_equal_score(self):
        values = [dict(utility=x) for x in [.3, .2, .3, .1, .3]]
        self.assertEqual(choose(values), 4)
        values[4]['utility'] = .29
        self.assertEqual(choose(values), 0)

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            choose([dict(utility=float('nan'))] * 5)

    def test_state_identity_preserves_original_timing_boundary(self):
        a = dict(results=[dict(test_id='public_0', status='WA', actual='0', expected='1', stderr='', seconds=.1)], wall_seconds=.1)
        b = json.loads(json.dumps(a))
        b['results'][0]['seconds'] = 2.
        self.assertEqual(state_key('print(0)', execution_from_dict(a), 0), state_key('print(0)', execution_from_dict(b), 0))

    def test_unicode_code_does_not_split_jsonl_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.jsonl'
            path.write_text(json.dumps(dict(code='x="\u2028"'), ensure_ascii=False) + '\n')
            self.assertEqual(len(list(lines(path))), 1)

    def test_preference_gradients_have_both_directions(self):
        import torch
        from losses import pair_loss
        rows = [dict(prediction_scope='whole_training_suite', problem_id='p', trajectory_id='t', test_id='s',
                     step_id=-1 if i == 0 else 0, verified_training_utility=value)
                for i, value in enumerate([.5, 1., 0., .5, .5])]
        values = torch.zeros(5, requires_grad=True)
        loss, signal = pair_loss(values, rows)
        loss.backward()
        self.assertEqual(signal, dict(pairs=2, candidate_better=1, incumbent_better=1))
        self.assertLess(float(values.grad[1]), 0)
        self.assertGreater(float(values.grad[2]), 0)
        self.assertEqual(float(values.grad[3]), 0)


if __name__ == '__main__':
    unittest.main()
