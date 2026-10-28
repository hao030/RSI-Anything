from collections import Counter, defaultdict
import random

def group_indices(rows):
    groups = defaultdict(list)
    seen = set()
    for i, row in enumerate(rows):
        identity = tuple((row[k] for k in ('problem_id', 'trajectory_id', 'step_id', 'candidate_id', 'test_id')))
        if identity in seen:
            raise ValueError('Duplicate transition identity: ' + repr(identity))
        seen.add(identity)
        groups[row['problem_id'], row['group_id']].append(i)
    return dict(groups)

def mixed(rows, indices):
    labels = {rows[i]['status'] == 0 for i in indices}
    return len(labels) == 2

def group_selection(rows, budget, seed, require_all_tasks=False, mixed_fraction=0.5, exact=False):
    if budget < 1 or not 0 <= mixed_fraction <= 1:
        raise ValueError('Invalid budget or contrast fraction')
    rng = random.Random(seed)
    groups = group_indices(rows)
    keys = list(groups)
    rng.shuffle(keys)
    priority = {key: j for j, key in enumerate(keys)}
    is_mixed = {key: mixed(rows, value) for key, value in groups.items()}
    selected, used, contrast = ([], 0, 0)
    per_task = Counter()

    def take(key):
        nonlocal used, contrast
        selected.append(key)
        used += len(groups[key])
        contrast += is_mixed[key]
        per_task[key[0]] += 1
    if require_all_tasks:
        for task in sorted({key[0] for key in keys}):
            candidates = [key for key in keys if key[0] == task]
            key = min(candidates, key=lambda k: (not is_mixed[k], len(groups[k]), priority[k]))
            if used + len(groups[key]) > budget:
                raise ValueError('Whole-group task coverage exceeds the declared label budget')
            take(key)
    remaining = set(keys) - set(selected)
    while remaining:
        eligible = [key for key in remaining if len(groups[key]) <= budget - used]
        if not eligible:
            break
        prefer_mixed = contrast < mixed_fraction * (len(selected) + 1)
        key = min(eligible, key=lambda k: (is_mixed[k] != prefer_mixed, per_task[k[0]], priority[k]))
        take(key)
        remaining.remove(key)
    if exact and used != budget:
        raise ValueError(f'Whole replay groups fill {used}/{budget}; no silent ratio change')
    indices = [i for key in selected for i in groups[key]]
    assert len(indices) == used == len(set(indices))
    return (indices, dict(selected_rows=used, selected_groups=len(selected), mixed_groups=contrast, different_tasks=len(per_task), requested_rows=budget, exact_budget_met=used == budget, candidate_groups_kept_whole=True, status_histogram=dict(Counter((rows[i]['status'] for i in indices)))))

def grouped_batches(rows, seed, tasks_per_batch=4):
    groups = group_indices(rows)
    by_task = defaultdict(list)
    for key in groups:
        by_task[key[0]].append(key)
    if len(by_task) < tasks_per_batch:
        raise ValueError('Not enough distinct tasks for a variance batch')
    rng = random.Random(seed)
    original = {task: list(keys) for task, keys in by_task.items()}
    queues = {task: list(keys) for task, keys in by_task.items()}
    for q in queues.values():
        rng.shuffle(q)
    is_mixed = {key: mixed(rows, values) for key, values in groups.items()}
    while any(queues.values()):
        active = [task for task, keys in queues.items() if keys]
        contrast_tasks = [task for task in active if any((is_mixed[k] for k in queues[task]))]
        chosen = [rng.choice(contrast_tasks)] if contrast_tasks else []
        candidates = [task for task in active if task not in chosen]
        chosen += rng.sample(candidates, min(tasks_per_batch - len(chosen), len(candidates)))
        if len(chosen) < tasks_per_batch:
            chosen += rng.sample([task for task in queues if task not in chosen], tasks_per_batch - len(chosen))
        indices, keys_used, padding = ([], [], [])
        for position, task in enumerate(chosen):
            q = queues[task]
            if q:
                want_mixed = position == 0 and bool(contrast_tasks)
                eligible = [key for key in q if is_mixed[key] == want_mixed]
                key = rng.choice(eligible or q)
                q.remove(key)
                padded = False
            else:
                key = rng.choice(original[task])
                padded = True
            indices.extend(groups[key])
            keys_used.append(key)
            if padded:
                padding.extend(groups[key])
        yield (indices, dict(group_keys=keys_used, padding_indices=padding, distinct_tasks=len(chosen), groups=len(keys_used)))

def geometry_scope(sample_count, distinct_tasks, dimension, minimum_samples=512, minimum_tasks=64):
    sufficient = sample_count >= max(minimum_samples, dimension + 1) and distinct_tasks >= minimum_tasks
    return dict(sample_count=sample_count, distinct_tasks=distinct_tasks, dimension=dimension, minimum_samples=max(minimum_samples, dimension + 1), minimum_tasks=minimum_tasks, sample_sufficient=sufficient, population_factor_claim='diagnostic_only' if sufficient else 'unassessed_insufficient_sample', geometry_pass_does_not_prove_task_performance=True)

def filter_whole_groups(rows, tokenizer, max_length):
    groups = group_indices(rows)
    invalid = {key for key, indices in groups.items() if any((max((len(tokenizer(rows[i][field])['input_ids']) for field in ('context', 'target'))) > max_length for i in indices))}
    keep = [i for key, indices in groups.items() if key not in invalid for i in indices]
    return ([rows[i] for i in keep], dict(input_rows=len(rows), input_groups=len(groups), overlength_groups_excluded=len(invalid), rows_excluded=len(rows) - len(keep), groupwise_filter=True))
