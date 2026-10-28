"""Build portable, deterministic release archives from a verified asset export."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def package(source, output, provenance_prefix, repository):
    source, output, repository = map(Path, (source, output, repository))
    if output.exists():
        raise FileExistsError(output)
    if not Path(provenance_prefix).is_absolute() or not provenance_prefix.rstrip('/'):
        raise ValueError('Provide the absolute original experiment root as provenance prefix')
    original = json.loads((source / 'manifest.json').read_text())
    stage = output / 'staging'
    stage.mkdir(parents=True)
    records = {}
    for name, info in original['files'].items():
        if name.startswith('models/'):
            continue
        src, dst = source / name, stage / name
        if src.stat().st_size != info['bytes'] or sha(src) != info['sha256']:
            raise ValueError('Original asset checksum mismatch: ' + name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith(('checkpoints/', 'training/')) and src.suffix in ('.json', '.jsonl', '.md'):
            text = src.read_text()
            # Only location metadata changes. Model inputs, labels and tensors are preserved.
            text = text.replace(provenance_prefix.rstrip('/') + '/', 'experiment-source/')
            if name.endswith('adapter_config.json'):
                config = json.loads(text)
                config['base_model_name_or_path'] = 'Qwen/Qwen2.5-Coder-3B-Instruct'
                text = json.dumps(config, indent=2) + '\n'
            if name.startswith('training/'):
                before, after = src.read_text().splitlines(), text.splitlines()
                if len(before) != len(after):
                    raise ValueError('Training row count changed: ' + name)
                for left, right in zip(before, after):
                    a, b = json.loads(left), json.loads(right)
                    a.pop('provenance', None)
                    b.pop('provenance', None)
                    if a != b:
                        raise ValueError('Computational training input changed: ' + name)
            dst.write_text(text)
        else:
            shutil.copyfile(src, dst)
        records[name] = {'original_sha256': info['sha256']}
    # Children first: a parent checkpoint hashes its nested checkpoint manifest too.
    for path in sorted((stage / 'checkpoints').rglob('manifest.json'), key=lambda p: len(p.parts), reverse=True):
        data = json.loads(path.read_text())
        data['files'] = {name: sha(path.parent / name) for name in data['files']}
        write(path, data)
    licenses = stage / 'licenses'
    licenses.mkdir()
    for src, name in [
        ('tasks/gui/licenses/Qwen2.5-Coder-3B-Instruct.LICENSE', 'Qwen-RESEARCH-LICENSE.txt'),
        ('docs/ASSET_CARD.md', 'ASSET_CARD.md'),
        ('NOTICE', 'CODE-NOTICE.txt'),
        ('LICENSE', 'CODE-Apache-2.0.txt'),
    ]:
        shutil.copyfile(repository / src, licenses / name)
        records['licenses/' + name] = {}
    groups = {'evaluation': {}, 'training': {}}
    for name, info in sorted(records.items()):
        path = stage / name
        info.update(bytes=path.stat().st_size, sha256=sha(path))
        group = 'training' if name.startswith(('training/', 'sources/', 'development/', 'checkpoints/initial/', 'checkpoints/semantic/', 'checkpoints/expanded/')) else 'evaluation'
        groups[group][name] = info
        if name.startswith('licenses/'):
            groups['training'][name] = info
    bundles = {}
    for group, files in groups.items():
        archive = output / (group + '.tar.gz')
        with archive.open('wb') as out, gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as gz, tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for name in files:
                info = tar.gettarinfo(str(stage / name), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with (stage / name).open('rb') as f:
                    tar.addfile(info, f)
        bundles[group] = dict(filename=archive.name, bytes=archive.stat().st_size, sha256=sha(archive), files=files)
    base = json.loads((repository / 'tasks/gui/src/rsi_anything_gui/assets/base_inputs.json').read_text())
    models = {}
    for role, label in [('world', 'Qwen2.5-Coder-3B-Instruct'), ('policy', 'ReasonFlux-Coder-7B')]:
        model = base[label]
        for name, record in model['files'].items():
            old = original['files']['models/' + role + '/' + name]
            if any(old[k] != record[k] for k in ('bytes', 'sha256')):
                raise ValueError('Base model identity mismatch: ' + name)
        models[role] = model
    catalog = dict(version=1, task='code_repair', release='assets-v1',
        experiment_sha256=original['experiment_sha256'],
        profiles={'evaluate': {'bundles': ['evaluation'], 'models': ['world']},
                  'full': {'bundles': ['evaluation', 'training'], 'models': ['world', 'policy']}},
        bundles=bundles, models=models,
        metadata_changes='Source locations use experiment-source identifiers; adapter base names use the official model ID; checkpoint manifests are rehashed. Model tensors and computational training inputs are unchanged.')
    write(output / 'code_repair.json', catalog)
    write(output / 'portable_training_sha256.json', {k: groups['training']['training/' + k + '.jsonl']['sha256'] for k in ('initial', 'semantic', 'expanded', 'incumbent')})
    print(json.dumps({'bundles': {k: {x: v[x] for x in ('bytes','sha256')} for k,v in bundles.items()}, 'files': len(records)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--provenance-prefix', required=True)
    p.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    package(a.source, a.output, a.provenance_prefix, a.repository)
