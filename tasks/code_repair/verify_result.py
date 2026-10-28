import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from artifacts import atomic_json, checkpoint_manifest, hash_object
from dataset import load_rows, tests_from_row
from evaluation import choose
from io_utils import PACKAGE, experiment, lines, read, sha, verify_assets


def audit(assets, output):
    assets, output = Path(assets).resolve(), Path(output).resolve()
    config = experiment()
    protocol = read(output / 'protocol.json')
    seal = read(output / 'public_seal.json')
    summary = read(output / 'summary.json')
    complete = read(output / 'complete.json')
    assert complete['status'] == 'COMPLETE'
    assert sha(output / 'summary.json') == complete['summary_sha256']
    assert sha(output / 'results.jsonl') == summary['results_sha256']
    assert sha(output / 'protocol.json') == summary['protocol_sha256'] == seal['protocol_sha256']
    assert protocol['experiment'] == config
    for relative, expected in protocol['source_sha256'].items():
        if Path(relative).name.startswith('._'):
            continue
        assert sha(PACKAGE / relative) == expected
    verify_assets(assets)
    _, identity = checkpoint_manifest(protocol['checkpoint'], formal=False)
    assert identity == protocol['world_identity']
    original = {(r['task_id'], seed): r for seed in range(3) for r in lines(assets / f'public/seed{seed}.jsonl')}
    public = {}
    for filename, digest in seal['files'].items():
        path = output / 'public' / filename
        assert sha(path) == digest
        for row in lines(path):
            key = (row['task_id'], row['seed'])
            assert key not in public
            public[key] = row
            item = original[key]
            assert row['world_identity'] == identity and row['initial_code'] == item['initial_code']
            assert not row['hidden_tests_accessed'] and row['optimizer_steps'] == 0
            assert len(row['trace']) <= 3
            current = item['initial_code']
            for step, trace in enumerate(row['trace']):
                assert trace['step'] == step and trace['current_code'] == current
                assert len(trace['candidates']) == 5
                assert trace['candidates'][4] == dict(candidate_id='keep_current', code=current)
                if step == 0:
                    assert trace['candidates'][:4] == [{k: c[k] for k in ['candidate_id', 'code']} for c in item['first_candidates']]
                assert trace['selected_index'] == choose(trace['predictions'])
                for prediction, candidate in zip(trace['predictions'], trace['candidates']):
                    assert prediction['candidate_id'] == candidate['candidate_id']
                    assert abs(prediction['utility'] - prediction['full_suite_pass_probability'] - .25 * prediction['predicted_pass_fraction']) < 1e-6
                current = trace['candidates'][trace['selected_index']]['code']
            assert row['final_code'] == current
    expected = {(tid, seed) for tid in config['evaluation']['tasks'] for seed in range(3)}
    assert len(public) == 600 and set(public) == expected
    official = load_rows(assets / 'judge/hidden.json')
    final = {}
    for row in lines(output / 'results.jsonl'):
        key = (row['task_id'], row['seed'])
        assert key not in final and all(row[k] == v for k, v in public[key].items())
        tests = tests_from_row(official[row['task_id']], 'hidden')
        timeout = max(3., min(float(official[row['task_id']].get('test_time_limit') or 1) * 2, 15.))
        cache_key = dict(executor_sha256=sha(PACKAGE / 'src/cure_wm.py'), code=row['final_code'],
            tests=[asdict(t) for t in tests], timeout=timeout, sandbox='required')
        cache_id = hash_object(cache_key)
        assert cache_id == row['hidden_execution_cache']
        path = output / 'judge_cache/hidden' / cache_id[:2] / (cache_id + '.json')
        entry = read(path)
        assert path.stat().st_mtime >= seal['time'] - 1
        assert entry['key'] == cache_key and entry['key_hash'] == cache_id and entry['value'] == row['final_hidden']
        outcomes = row['final_hidden']['results']
        assert [r['test_id'] for r in outcomes] == [t.test_id for t in tests]
        assert [r['expected'] for r in outcomes] == [t.expected for t in tests]
        success = bool(outcomes) and all(r['status'] == 'AC' for r in outcomes)
        assert success == row['hidden_allpass']
        final[key] = success
    assert len(final) == 600 and set(final) == expected
    counts = [sum(v for (tid, s), v in final.items() if s == seed) for seed in range(3)]
    assert summary['per_seed_successes'] == counts and summary['successes'] == sum(counts)
    reference = {(r['task_id'], r['seed']): r for r in lines(assets / 'reference/decisions.jsonl')}
    matches = sum(hashlib.sha256(r['final_code'].encode()).hexdigest() == reference[k]['final_code_sha256']
        and [t['selected_index'] for t in r['trace']] == reference[k]['choices'] for k, r in public.items())
    assert matches == summary['reference_decisions_matched']
    report = dict(status='PASS', records=600, successes=sum(counts), rate=sum(counts) / 600,
        per_seed_successes=counts, reference_decisions_matched=matches,
        reference_counts_matched=counts == config['reference']['per_seed_successes'],
        all_original_hidden_fixtures_verified=True, fresh_hidden_cache_verified=True,
        native_checkpoint_identity=identity, public_seal_sha256=sha(output / 'public_seal.json'),
        summary_sha256=sha(output / 'summary.json'), results_sha256=sha(output / 'results.jsonl'))
    if (output / 'independent_audit.json').exists():
        assert read(output / 'independent_audit.json') == report
    else:
        atomic_json(output / 'independent_audit.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.assets, args.output)
