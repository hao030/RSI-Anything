import collections
import torch
from torch.nn import functional as F
from semantic_world import GLOBAL

def pair_loss(scores, rows):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r['prediction_scope'] == GLOBAL:
            groups[tuple((r[k] for k in ['problem_id', 'trajectory_id', 'test_id']))].append(i)
    values = []
    weights = []
    positive = negative = 0
    for ids in groups.values():
        initial = [i for i in ids if rows[i]['step_id'] == -1]
        candidates = [i for i in ids if rows[i]['step_id'] == 0]
        if not candidates:
            continue
        assert len(initial) == 1 and len(candidates) == 4
        current = initial[0]
        for i in candidates:
            difference = rows[i]['verified_training_utility'] - rows[current]['verified_training_utility']
            if difference == 0:
                continue
            sign = 1.0 if difference > 0 else -1.0
            values.append(0.1 * F.softplus(-sign * (scores[i] - scores[current]) / 0.1))
            weights.append(abs(difference) / 1.25)
            positive += difference > 0
            negative += difference < 0
    if not values:
        return (scores.sum() * 0, dict(pairs=0, candidate_better=0, incumbent_better=0))
    weights = scores.new_tensor(weights)
    return ((torch.stack(values) * weights).sum() / weights.sum(), dict(pairs=len(values), candidate_better=positive, incumbent_better=negative))

def objective(model, teacher, rows, pair_weight, anchor_weight):
    loss, parts, contexts, factors, signal = model.training_loss(rows, teacher.public_world, 1.0)
    latent, _ = model.public_world.predict_latent(contexts)
    out = model.suite_readout(latent)
    probabilities = F.softmax(out[:, :5], dim=-1)
    fraction = torch.sigmoid(out[:, 6])
    scores = probabilities[:, 0] + 0.25 * fraction
    relative, pair_signal = pair_loss(scores, rows)
    anchor = out.sum() * 0
    ids = [i for i, r in enumerate(rows) if r['prediction_scope'] == GLOBAL]
    if anchor_weight and ids:
        with torch.no_grad():
            reference = teacher.score_suite([rows[i]['context'] for i in ids])
        anchor = F.kl_div(F.log_softmax(out[ids, :5], dim=-1), reference['failure_status_probabilities'], reduction='batchmean')
        anchor = anchor + F.mse_loss(fraction[ids], reference['predicted_pass_fraction'])
    total = loss + 0.8 * parts['listwise'] + pair_weight * relative + anchor_weight * anchor
    return (total, dict(parts, incumbent_preference=relative, suite_anchor=anchor), dict(signal, **pair_signal))
