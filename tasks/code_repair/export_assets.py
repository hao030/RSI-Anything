import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from io_utils import experiment, lines, read, sha, write_lines
from artifacts import atomic_json, hash_object


def export(source, output, mode):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    r, old = source / 'rsi_jepa_g6', source / 'cure_wm_lite'
    q = r / 'evaluations/historical_jepa_200_optimization_20261006'
    first = r / 'evaluations/historical_jepa_200_20261006'
    final = q / 'same_200_retest_incumbent_round4'
    output.mkdir(parents=True)
    records = {}

    def include(relative, original):
        original = Path(original).resolve()
        if original.is_dir():
            for child in sorted(original.rglob('*')):
                if child.is_file() and '__pycache__' not in child.parts:
                    include(str(Path(relative) / child.relative_to(original)), child)
            return
        if not original.is_file():
            raise FileNotFoundError(original)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == 'symlink':
            target.symlink_to(original)
        else:
            shutil.copy2(original, target)
        records[relative] = dict(bytes=original.stat().st_size, sha256=sha(original), source=str(original))

    def generated(relative, values):
        target = output / relative
        if relative.endswith('.jsonl'):
            write_lines(target, values)
        else:
            atomic_json(target, values)
        records[relative] = dict(bytes=target.stat().st_size, sha256=sha(target), source='derived_from_verified_public_inputs')

    mapping = {
        'models/world': old / 'models/Qwen2.5-Coder-3B-Instruct',
        'models/policy': old / 'models/ReasonFlux-Coder-7B',
        'checkpoints/initial': r / 'checkpoints/initial_opf_seed0',
        'checkpoints/semantic': q / 'fits/suite_initial_seed1_distill0p25/joint_checkpoint',
        'checkpoints/expanded': q / 'expanded_fits_round2/expanded_seed1_distill1p0_rank1p0/joint_checkpoint',
        'checkpoints/incumbent': q / 'incumbent_aware_update_v1/fits/seed0_pair4_anchor0/joint_checkpoint',
        'training/initial.jsonl': r / 'data/initial_world_train.jsonl',
        'training/semantic.jsonl': q / 'prepared_seed0/training.jsonl',
        'training/expanded.jsonl': q / 'expanded_training_v1/training.jsonl',
        'training/incumbent.jsonl': q / 'incumbent_aware_update_v1/data/training.jsonl',
        'sources/g2_suite.jsonl': r / 'repairs/world_update_v3_global_quality/data/g2_whole_training_suite_with_initial_incumbents.jsonl',
        'sources/legacy_suite.jsonl': q / 'legacy_training_suite_2000_v1/training_transitions.jsonl',
        'development/public_cases.jsonl': q / 'development_300_v2/public_cases.jsonl',
        'development/observed_quality.jsonl': q / 'development_300_v2/observed_quality.jsonl',
    }
    for seed in [0, 1, 2]:
        mapping[f'public/seed{seed}.jsonl'] = first / f'public/seed{seed}.jsonl'
    for relative, original in mapping.items():
        include(relative, original)
    for stage, expected in experiment()['training_sha256'].items():
        assert records[f'training/{stage}.jsonl']['sha256'] == expected

    parents = [first, q / 'same_200_retest_round1', q / 'same_200_retest_guard_round2',
               q / 'same_200_retest_expanded_round3', final]
    needed = set()
    reference = []
    for path in sorted((final / 'public_results').glob('*.jsonl')):
        for row in lines(path):
            assert not row['hidden_tests_accessed'] and row['online_optimizer_steps'] == 0
            reference.append(dict(task_id=row['task_id'], seed=row['seed'],
                final_code_sha256=__import__('hashlib').sha256(row['final_code'].encode()).hexdigest(),
                choices=[step['selected_index'] for step in row['trace']]))
            for step in row['trace']:
                for namespace, field in [('historical_generation', 'generation_cache'), ('historical_visible', 'execution_cache')]:
                    identity = step[field].get('cache_identity')
                    if identity:
                        needed.add((namespace, identity))
    assert len(reference) == 600
    for namespace, identity in sorted(needed):
        rel = Path(namespace) / identity[:2] / (identity + '.json')
        original = next((p / 'cache' / rel for p in parents if (p / 'cache' / rel).exists()), None)
        if original is None:
            raise FileNotFoundError(str(rel))
        entry = read(original)
        assert entry['key_hash'] == identity == hash_object(entry['key'])
        generated(str(Path('public_cache') / rel), dict(key=entry['key'], key_hash=identity, value=entry['value']))
    generated('reference/decisions.jsonl', reference)
    tasks = set(experiment()['evaluation']['tasks'])
    eval_path = old / 'data/CodeContests/test/CodeContests.json'
    hidden = [row for row in read(eval_path) if str(row['task_id']) in tasks]
    assert len(hidden) == 200
    generated('judge/hidden.json', hidden)
    records['judge/hidden.json']['source'] = str(eval_path)
    records['judge/hidden.json']['original_source_sha256'] = sha(eval_path)
    atomic_json(output / 'manifest.json', dict(version=1, files=records,
        experiment_sha256=sha(Path(__file__).parent / 'config/experiment.json'),
        recorded_public_inputs=True, hidden_execution_results_included=False))
    print(json.dumps(dict(files=len(records), output=str(output), mode=mode)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['copy', 'symlink'], default='copy')
    a = p.parse_args()
    export(a.source_root, a.output, a.mode)
