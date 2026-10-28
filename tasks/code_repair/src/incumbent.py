import collections
import random
from grouping import group_indices

def state_signature(row):
    before, sep, tail = row['context'].rpartition('\nCandidate action:\n')
    action, sep2, tests = tail.rpartition('\nOfficial public test suite:\n')
    assert sep and sep2 and before and action
    return (before, tests)

def make_units(rows):
    groups = group_indices(rows)
    initial = {}
    first = {}
    for key, ids in groups.items():
        sample = rows[ids[0]]
        if sample['prediction_scope'] != 'whole_training_suite':
            continue
        identity = tuple((sample[k] for k in ['problem_id', 'trajectory_id', 'test_id']))
        if sample['step_id'] == -1:
            assert len(ids) == 1 and sample['candidate_id'] == 'keep_initial' and (identity not in initial)
            initial[identity] = key
        else:
            assert sample['step_id'] == 0 and len(ids) == 4 and (identity not in first)
            first[identity] = key
    consumed = set()
    units = []
    for identity, key in sorted(first.items()):
        current_key = initial[identity]
        current_id = groups[current_key][0]
        candidates = groups[key]
        assert all((state_signature(rows[i]) == state_signature(rows[current_id]) for i in candidates))
        ids = candidates + [current_id]
        values = [rows[i]['verified_training_utility'] for i in ids]
        units.append(dict(problem_id=identity[0], indices=ids, incumbent=current_id, candidates=candidates, contrast=max(values) > min(values), group_keys=[list(key), list(current_key)]))
        consumed.update([key, current_key])
    for key, ids in sorted(groups.items()):
        if key not in consumed:
            units.append(dict(problem_id=key[0], indices=ids, incumbent=None, candidates=[], contrast=len({rows[i]['status'] == 0 for i in ids}) > 1, group_keys=[list(key)]))
    counts = collections.Counter((i for u in units for i in u['indices']))
    assert counts == collections.Counter(range(len(rows)))
    return units

def batches_for(units, seed):
    rng = random.Random(seed)
    queues = collections.defaultdict(list)
    for i, u in enumerate(units):
        queues[u['problem_id']].append(i)
    original = {k: list(v) for k, v in queues.items()}
    for q in queues.values():
        rng.shuffle(q)
    output = []
    while any(queues.values()):
        active = [task for task, q in queues.items() if q]
        contrast = [task for task in active if any((units[i]['contrast'] for i in queues[task]))]
        chosen = [rng.choice(contrast)] if contrast else []
        remaining = [task for task in active if task not in chosen]
        chosen += rng.sample(remaining, min(4 - len(chosen), len(remaining)))
        if len(chosen) < 4:
            chosen += rng.sample([task for task in queues if task not in chosen], 4 - len(chosen))
        ids = []
        padding = []
        selected = []
        for position, task in enumerate(chosen):
            q = queues[task]
            if q:
                possible = [i for i in q if units[i]['contrast'] == (position == 0 and bool(contrast))]
                i = rng.choice(possible or q)
                q.remove(i)
                padded = False
            else:
                i = rng.choice(original[task])
                padded = True
            ids.extend(units[i]['indices'])
            selected.append(i)
            if padded:
                padding.extend(units[i]['indices'])
        output.append(dict(indices=ids, padding_indices=padding, unit_ids=selected, distinct_tasks=4))
    return output
