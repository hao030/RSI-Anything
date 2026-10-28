import hashlib
import json
import os
from pathlib import Path
import time

from .settings import ROOT, PORT, BASE_WORLD, BASE_POLICY, source_hashes, runtime_identity, verify_frozen
verify_frozen()
for name in ['data','checkpoints','results','logs','screenshots']:
    (ROOT/name).mkdir(parents=True, exist_ok=True)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)

def read_jsonl(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')

def context(obs, action):
    return json.dumps({'instruction':obs['instruction'], 'visible_elements':obs['elements'], 'remaining_actions':obs['remaining_actions'], 'candidate_action':action}, sort_keys=True, ensure_ascii=False)

def target(obs):
    return json.dumps({'instruction':obs['instruction'],'visible_elements':obs['elements'], 'terminated':obs['done'],
                       'submitted':obs['submitted'],'remaining_actions':obs['remaining_actions']}, sort_keys=True, ensure_ascii=False)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomic_jsonl(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(p)

def checkpoint_identity(path):
    if path is None:return 'base'
    p=Path(path)
    meta=p/'training.json'
    weights=p/'gui_state.pt' if (p/'gui_state.pt').exists() else p/'adapter_model.safetensors'
    return digest([str(p.resolve()),file_hash(meta),file_hash(weights)])

def effective_cost(rows):
    return sum(r['policy_nominal_seconds']+r['world_seconds']+r['browser_seconds'] for r in rows)/len(rows)

def goal_bits(obs):
    sentence = obs['instruction']
    prefix, suffix = 'Select ', ' and click Submit.'
    if not sentence.startswith(prefix) or not sentence.endswith(suffix):
        raise ValueError('Unexpected official instruction: '+sentence)
    wanted = sentence[len(prefix):-len(suffix)]
    names = set() if wanted == 'nothing' else set(wanted.split(', '))
    labels = [x['label'] for x in obs['elements']]
    if not names.issubset(set(labels)):
        raise ValueError('Target names absent from visible labels')
    return [x in names for x in labels]

def correctness(obs):
    desired = goal_bits(obs)
    return sum(e['checked']==g for e,g in zip(obs['elements'], desired))/len(desired)

def legal_actions(obs):
    return [{'kind':'click','slot':e['slot']} for e in obs['elements']]+[{'kind':'submit'}]

def label_transition(before, action, after, seconds, episode_id, step):
    success = bool(after['done'] and after.get('submitted') and correctness(after)==1.)
    utility = (1. if success else -1.) if after['done'] else correctness(after)-correctness(before)
    return {'episode_id':episode_id, 'step':step, 'context':context(before,action), 'target':target(after),
            'before':before, 'action':action, 'after':after, 'checked':[int(e['checked']) for e in after['elements']],
            'done':int(after['done']), 'success':int(success), 'utility':utility,
            'seconds':seconds, 'group_id':digest([before['instruction'],before['elements'],before['remaining_actions']]), 'pair_hash':digest([context(before,action),target(after)])}
