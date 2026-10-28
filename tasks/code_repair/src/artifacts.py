import json, hashlib, fcntl, os, time
from pathlib import Path
ROOT = Path.cwd() / 'runs'

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def hash_object(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temp.open('w') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)

def checkpoint_manifest(path, formal=True):
    path = Path(path)
    m = json.load(open(path / 'manifest.json'))
    if formal and (not m.get('formal_checkpoint', False)):
        raise ValueError('Preflight checkpoint is forbidden in formal experiments')
    for name, expected in m['files'].items():
        p = path / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'checkpoint integrity failure: {p}')
    return (m, hash_object(m))

class Cache:

    def __init__(self, root=ROOT / 'cache'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, namespace, key, produce):
        digest = hash_object(key)
        folder = self.root / namespace / digest[:2]
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / (digest + '.json')
        with (folder / (digest + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if p.exists():
                value = json.load(open(p))
                assert value['key_hash'] == digest
                return (value['value'], True, digest)
            value = produce()
            atomic_json(p, dict(key_hash=digest, key=key, value=value, time=time.time()))
            return (value, False, digest)
