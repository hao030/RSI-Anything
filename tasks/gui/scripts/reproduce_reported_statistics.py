"""Recompute original paired comparisons from sanitized formal episode endpoints."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor/jepa-anything-core/src')]
from rsi_anything_gui.statistics import paired


def main():
    records = json.loads((ROOT / 'docs/reported_episode_endpoints.json').read_text())
    expected = json.loads((ROOT / 'docs/reported_results.json').read_text())
    with tempfile.TemporaryDirectory(prefix='rsi-statistics-') as directory:
        paths = {}
        for name, rows in records.items():
            p = Path(directory) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            paths[name] = p

        def compare(a, b, target):
            actual = paired([paths[x] for x in a], [paths[x] for x in b])
            for field in ['effect', 'task_cluster_bootstrap_95_ci', 'seed_effects']:
                if not np.allclose(actual[field], target[field], rtol=0, atol=1e-12):
                    raise RuntimeError('Reported statistic differs: ' + field)
            if actual['per_run_tests'] != target['per_run_tests']:
                raise RuntimeError('Exact paired test differs')
            return actual

        results = {}
        for kind, label in [('pure', 'primary_pure'), ('selected', 'selected_update_effect')]:
            results[label] = compare([f'opf_seed{s}/updated_{kind}_test.jsonl' for s in range(3)],
                                     [f'opf_seed{s}/frozen_{kind}_test.jsonl' for s in range(3)], expected[label])
        results['confirmation'] = compare([f'opf_seed{s}/updated_confirmation.jsonl' for s in range(3)],
                                           [f'opf_seed{s}/frozen_confirmation.jsonl' for s in range(3)], expected['confirmation'])
        secondary = [
            ('pure_updated_vs_policy', 'opf_seed0/updated_pure_test.jsonl', 'baseline_policy_test.jsonl'),
            ('selected_updated_vs_policy', 'opf_seed0/updated_selected_test.jsonl', 'baseline_policy_test.jsonl'),
            ('pure_opf_vs_ce', 'opf_seed0/updated_pure_test.jsonl', 'ce_seed0/updated_pure_test.jsonl'),
            ('pure_opf_vs_standard', 'opf_seed0/updated_pure_test.jsonl', 'standard_seed0/updated_pure_test.jsonl'),
            ('initial_opf_vs_ce_shared_data', 'opf_seed0/frozen_pure_test.jsonl', 'ce_seed0/frozen_pure_test.jsonl'),
            ('initial_opf_vs_standard_shared_data', 'opf_seed0/frozen_pure_test.jsonl', 'standard_seed0/frozen_pure_test.jsonl')]
        for name, a, b in secondary:
            results[name] = compare([a], [b], expected['secondary'][name])
        previous = 0.
        for i, (p, name) in enumerate(sorted((results[n]['per_run_tests'][0]['exact_mcnemar_p'], n) for n, _, _ in secondary)):
            previous = max(previous, min(1., p * (len(secondary) - i)))
            assert abs(previous - expected['secondary'][name]['holm_adjusted_p']) < 1e-12
        for generation, reference in enumerate(expected['rsi'], 1):
            results[f'rsi_{generation}_policy'] = compare([f'rsi_g{generation}/new_policy_test.jsonl'],
                                                        [f'rsi_g{generation}/old_policy_test.jsonl'], reference['paired_policy_only'])
            results[f'rsi_{generation}_selected'] = compare([f'rsi_g{generation}/new_new_selected_test.jsonl'],
                                                          [f'rsi_g{generation}/old_old_selected_test.jsonl'], reference['paired_joint_selected'])
    report = {'passed': True, 'comparisons': len(results), 'formal_endpoint_records': sum(map(len, records.values())),
              'mean_confirmation_gain_pp': results['confirmation']['effect'] * 100,
              'model_weights_or_full_traces_verified': False}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
