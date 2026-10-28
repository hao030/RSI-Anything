"""Regenerate reviewed manifests explicitly; never stage files automatically."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update_gui():
    gui = ROOT / 'tasks/gui'
    spec = importlib.util.spec_from_file_location('gui_release_verifier', gui / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_path = gui / 'release_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = list(module.release_files())
    for path in files:
        module.check_private_material(path)
    manifest['files'] = {str(p.relative_to(gui)): {'bytes': p.stat().st_size,
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    module.main()


def update_repository():
    # Hash the Git index so unrelated unstaged changes cannot enter a commit's manifest.
    output = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--stage', '-z'])
    entries = []
    for record in output.split(b'\0'):
        if not record:
            continue
        metadata, path = record.split(b'\t', 1)
        mode, object_id, stage = metadata.split()
        if stage != b'0':
            raise RuntimeError('Resolve index conflicts before generating the manifest')
        if path != b'SHA256SUMS.json':
            entries.append((path.decode(), object_id))
    if not entries:
        raise RuntimeError('Stage the intended source files first')
    data = subprocess.check_output(['git', '-C', str(ROOT), 'cat-file', '--batch'],
                                   input=b''.join(sha + b'\n' for _, sha in entries))
    offset = 0
    manifest = {}
    for path, expected in entries:
        end = data.index(b'\n', offset)
        object_id, kind, size = data[offset:end].split()
        if object_id != expected or kind != b'blob':
            raise RuntimeError('Unexpected staged Git object: ' + path)
        start = end + 1
        content = data[start:start + int(size)]
        manifest[path] = hashlib.sha256(content).hexdigest()
        offset = start + int(size) + 1
    (ROOT / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'updated': 'SHA256SUMS.json', 'staged_files': len(manifest),
                      'next': 'Review and stage SHA256SUMS.json'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--gui', action='store_true')
    mode.add_argument('--repository', action='store_true')
    args = parser.parse_args()
    update_gui() if args.gui else update_repository()
