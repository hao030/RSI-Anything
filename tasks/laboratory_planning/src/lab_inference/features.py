"""Deterministic, label-free task/candidate features and hard legality checks."""
from __future__ import annotations
import ast, hashlib, json, re, sys
import numpy as np
from .plan_schema import code
TASK_DIM = 128
CANDIDATE_DIM = 128
STRUCTURE_DIM = 16
FEATURE_DIM = TASK_DIM + CANDIDATE_DIM + STRUCTURE_DIM
INPUT_DIM = FEATURE_DIM * 2

def hashed(text, dim):
    a = np.zeros(dim, dtype=np.float32)
    for token in re.findall('[A-Za-z_][A-Za-z_0-9]*|\\d+(?:\\.\\d+)?', text.lower()):
        b = hashlib.blake2b(token.encode(), digest_size=8).digest()
        a[int.from_bytes(b[:4], 'little') % dim] += 1 if b[4] & 1 else -1
    a = np.sign(a) * np.log1p(np.abs(a))
    return a / max(float(np.linalg.norm(a)), 1.0)

def task_features(task):
    from .plan_schema import FIELDS
    if set(task) != FIELDS:
        raise ValueError('Feature input has label-bearing fields')
    text = ' '.join((str(task[k]) for k in ('context', 'goal', 'constraints', 'available_inputs', 'action_pool')))
    return hashed(text, TASK_DIM)

def structure(task, completion):
    from .plan_schema import is_valid_sequence, raw_input_names
    from .plan_schema import parse_steps, extract_python_steps
    vals = np.zeros(STRUCTURE_DIM, dtype=np.float32)
    legal = False
    try:
        tree = ast.parse(code(completion))
        signatures = {n.name: n for n in ast.parse(task['action_pool']).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        steps = parse_steps(extract_python_steps(completion))
        valid, _ = is_valid_sequence(steps, raw_inputs=raw_input_names(task['available_inputs']), action_pool_names=set(signatures))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        bad_keyword = missing_required = positional = 0
        for c in calls:
            positional += len(c.args)
            if not isinstance(c.func, ast.Name) or c.func.id not in signatures:
                bad_keyword += 1
                continue
            sig = signatures[c.func.id]
            parameters = sig.args.posonlyargs + sig.args.args
            names = {n.arg for n in parameters + sig.args.kwonlyargs}
            required = {n.arg for n in parameters[:len(parameters) - len(sig.args.defaults)]}
            required |= {n.arg for n, d in zip(sig.args.kwonlyargs, sig.args.kw_defaults) if d is None}
            provided = {k.arg for k in c.keywords}
            bad_keyword += len(provided - names)
            missing_required += len(required - provided)
        legal = bool(steps) and bool(valid) and (not (bad_keyword or missing_required or positional))
        actions = [s.action for s in steps]
        kw = [k for c in calls for k in c.keywords]
        vals[:] = [1, int(valid), int(legal), len(steps) / 13, len(set(actions)) / 13, (len(actions) - len(set(actions))) / 13, len(kw) / 50, len([n for n in ast.walk(tree) if isinstance(n, ast.Constant)]) / 50, len([n for n in ast.walk(tree) if isinstance(n, ast.Name)]) / 50, len([s for s in steps if s.output]) / 13, len(signatures) / 50, len(raw_input_names(task['available_inputs'])) / 30, bad_keyword / 13, missing_required / 13, positional / 13, len(completion) / 5000]
    except (ValueError, SyntaxError, TypeError, KeyError):
        vals[11] = len(completion) / 5000
    return (np.clip(vals, 0, 5), legal)

def candidate_features(task, completion):
    v, legal = structure(task, completion)
    return (np.concatenate([task_features(task), hashed(completion, CANDIDATE_DIM), v]), legal)

def relative_features(task, candidates):
    f = []
    legal = []
    for c in candidates:
        x, l = candidate_features(task, c['completion'])
        f.append(x)
        legal.append(l)
    f = np.array(f, dtype=np.float32)
    return (np.concatenate([f - f[0], np.repeat(f[:1], len(f), axis=0)], axis=1), np.array(legal, dtype=bool))
