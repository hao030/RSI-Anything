import collections
from pathlib import Path

from artifacts import atomic_json
from io_utils import lines


def evaluate(model, assets, output):
    cases = list(lines(Path(assets) / 'development/public_cases.jsonl'))
    predictions = []
    model.eval()
    for row in cases:
        value = model.score_suite([row['context']])
        predictions.append(dict(case_id=row['case_id'], task_id=row['task_id'], step=row['step'],
            utility=float(value['utility'][0]), full_suite_pass_probability=float(value['full_suite_pass_probability'][0]),
            predicted_pass_fraction=float(value['predicted_pass_fraction'][0])))
    labels = {row['case_id']: row for row in lines(Path(assets) / 'development/observed_quality.jsonl')}
    assert len(predictions) == len(labels) == 2440
    by_id = {row['case_id']: row for row in predictions}
    initial, groups = {}, collections.defaultdict(list)
    for row in predictions:
        if row['step'] == -1:
            initial[row['task_id']] = row['case_id']
        else:
            groups[(row['task_id'], row['step'])].append(row['case_id'])
    assert len(initial) == 300 and len(groups) == 535
    chosen = {key: max(ids, key=lambda cid: by_id[cid]['utility']) for key, ids in groups.items()}
    original = guarded = 0
    for tid, current in initial.items():
        candidate = chosen.get((tid, 0), current)
        protected = candidate if by_id[candidate]['utility'] > by_id[current]['utility'] else current
        original += bool(labels[candidate]['suite_allpass'])
        guarded += bool(labels[protected]['suite_allpass'])
    report = dict(predictions=predictions, original_successes=original, guarded_successes=guarded,
        tasks=300, mean_selected_observed_utility=sum(labels[cid]['utility'] for cid in chosen.values()) / 535,
        full_suite_probability_brier=sum((row['full_suite_pass_probability'] - labels[row['case_id']]['suite_allpass']) ** 2
                                        for row in predictions) / 2440,
        scope='Recorded first decisions for development selection')
    atomic_json(output, report)
    return report
