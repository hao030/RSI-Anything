import collections
from pathlib import Path

from artifacts import atomic_json
from grouping import filter_whole_groups, group_selection, grouped_batches
from incumbent import make_units, batches_for
from io_utils import experiment, lines, sha, verify_assets, verify_training_hash, write_lines


def prepare(args):
    from transformers import AutoTokenizer
    assets, output = Path(args.assets).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    verify_assets(assets, prefix=['sources/', 'training/initial.jsonl', 'models/world/'])
    initial = list(lines(assets / 'training/initial.jsonl'))
    public = [dict(row, prediction_scope='per_public_test') for row in initial]
    g2 = [dict(row, prediction_scope='whole_training_suite') for row in lines(assets / 'sources/g2_suite.jsonl')]
    legacy = list(lines(assets / 'sources/legacy_suite.jsonl'))
    tokenizer = AutoTokenizer.from_pretrained(assets / 'models/world', local_files_only=True)
    public, _ = filter_whole_groups(public, tokenizer, 16384)
    g2, _ = filter_whole_groups(g2, tokenizer, 16384)
    selected, _ = group_selection(public, len(g2), 1, exact=True)
    semantic = g2 + [public[i] for i in selected]
    global_rows = sorted(legacy + g2, key=lambda row: (
        str(row['problem_id']), row['trajectory_id'], row['step_id'], row['candidate_id'], row['test_id']))
    global_rows, _ = filter_whole_groups(global_rows, tokenizer, 16384)
    selected, _ = group_selection(public, len(global_rows), 20261006, exact=True)
    expanded = global_rows + [public[i] for i in selected]
    first = [row for row in expanded if row['prediction_scope'] == 'whole_training_suite' and row['step_id'] in [-1, 0]]
    replay = [row for row in expanded if row['prediction_scope'] == 'per_public_test']
    selected, _ = group_selection(replay, len(first), 20261007, exact=True)
    incumbent = first + [replay[i] for i in selected]
    datasets = dict(semantic=semantic, expanded=expanded, incumbent=incumbent)
    config = experiment()
    output.mkdir(parents=True)
    report = {}
    for name, rows in datasets.items():
        path = output / f'{name}.jsonl'
        write_lines(path, rows)
        assert len(rows) == config['training'][name]['rows']
        verify_training_hash(name, path)
        seed = config['training'][name]['seed']
        if name == 'incumbent':
            units = make_units(rows)
            batches = batches_for(units, seed)
            assert sum(unit['incumbent'] is not None for unit in units) == 2174
        else:
            batches = [dict(indices=ids, **info) for ids, info in grouped_batches(rows, seed)]
        counts = collections.Counter(i for batch in batches for i in batch['indices'] if i not in batch['padding_indices'])
        assert counts == collections.Counter(range(len(rows)))
        atomic_json(output / f'{name}.batches.json', batches)
        report[name] = dict(rows=len(rows), sha256=sha(path), batches=len(batches))
    atomic_json(output / 'complete.json', report)
    print(report, flush=True)
