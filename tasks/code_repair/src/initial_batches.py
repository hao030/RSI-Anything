import random
from collections import defaultdict

def batch_indices(rows, seed, batch_tasks=4):
    rng = random.Random(seed)
    tasks = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        tasks[r['problem_id']][r['group_id']].append(i)
    queues = {}
    for tid, groups in tasks.items():
        units = []
        for group in groups.values():
            rng.shuffle(group)
            units.extend((group[i:i + 2] for i in range(0, len(group), 2)))
        rng.shuffle(units)
        queues[tid] = units
    all_tasks = sorted(queues)
    if len(all_tasks) < batch_tasks:
        raise RuntimeError('Need at least four distinct tasks for variance batch')
    while any(queues.values()):
        active = [tid for tid in all_tasks if queues[tid]]
        chosen = rng.sample(active, min(batch_tasks, len(active)))
        if len(chosen) < batch_tasks:
            chosen += rng.sample([t for t in all_tasks if t not in chosen], batch_tasks - len(chosen))
        batch = []
        for tid in chosen:
            if queues[tid]:
                batch.extend(queues[tid].pop())
            else:
                batch.append(rng.choice([i for group in tasks[tid].values() for i in group]))
        yield batch
