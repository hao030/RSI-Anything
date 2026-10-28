"""Acquire external model inputs without distributing their weights."""
import json
from .settings import PACKAGE, BASE_WORLD, BASE_POLICY, ROOT, _hash


def prepare_models(download=False):
    records = json.loads((PACKAGE / 'assets/base_inputs.json').read_text())
    for directory in [BASE_WORLD, BASE_POLICY]:
        record = records[directory.name]
        if download:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=record['repo_id'], revision=record['revision'], local_dir=str(directory))
        problems = []
        for name, info in record['files'].items():
            path = directory / name
            if not path.is_file():
                problems.append(name + ': missing')
            elif path.stat().st_size != info['bytes'] or _hash(path) != info['sha256']:
                problems.append(name + ': hash/size mismatch')
        if problems:
            raise RuntimeError(directory.name + '\n' + '\n'.join(problems))
        print(directory.name + ': exact static inputs verified', flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
