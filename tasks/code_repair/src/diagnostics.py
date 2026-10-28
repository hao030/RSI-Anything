import collections
import torch
from grouping import geometry_scope
from world import D

def geometry(model, rows):
    by_task = collections.defaultdict(list)
    for i, row in enumerate(rows):
        by_task[row['problem_id']].append(i)
    chosen = []
    for depth in range(max(map(len, by_task.values()))):
        for task in sorted(by_task):
            if depth < len(by_task[task]):
                chosen.append(by_task[task][depth])
            if len(chosen) == 512:
                break
        if len(chosen) == 512:
            break
    scope = geometry_scope(len(chosen), len({rows[i]['problem_id'] for i in chosen}), D)
    assert scope['sample_sufficient']
    model.eval()
    with torch.no_grad():
        contexts = model.public_world.encode([rows[i]['context'] for i in chosen])
        report = model.public_world.geometry(contexts)
    return dict(scope=scope, indices=chosen, native_geometry=report, numerical_results_preserved_without_relaxing_thresholds=True)

def delta(model, before):
    prefixes = ['public_world.online.backbone', 'public_world.online.projection', 'public_world.predictor', 'public_world.opf', 'public_world.readout', 'suite_readout', 'public_world.target']
    changes = {name: float((parameter.detach().float().cpu() - before[name]).norm()) for name, parameter in model.named_parameters() if name in before}
    return {prefix: sum((v for name, v in changes.items() if name.startswith(prefix))) for prefix in prefixes}
