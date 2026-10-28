import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def lines(path):
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError(f'Incomplete JSONL record: {path}')
            yield json.loads(line)


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def experiment():
    return read(PACKAGE / 'config/experiment.json')


def verify_assets(root, prefix=None):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    checked = 0
    for name, record in manifest['files'].items():
        if prefix is not None and not any(name.startswith(p) for p in prefix):
            continue
        path = root / name
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Asset integrity failure: {name}')
        checked += 1
    return dict(checked_files=checked, manifest_sha256=sha(root / 'manifest.json'))


def verify_training_hash(stage, path):
    config = experiment()
    valid = {config['training_sha256'][stage]}
    valid.update([config.get('portable_training_sha256', {}).get(stage)])
    if sha(Path(path)) not in valid:
        raise ValueError('Training input identity mismatch: ' + stage)
