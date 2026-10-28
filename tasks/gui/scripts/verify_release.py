"""Check a prepared source release's integrity and accidental private material."""
import hashlib
import ipaddress
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'build', 'dist', 'runs', '.venv'}


def release_files():
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(x in EXCLUDED or x.endswith('.egg-info') for x in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError('Release contains a symbolic link: ' + str(relative))
        if path.is_file() and path.name not in ['release_manifest.json', 'PKG-INFO'] and path.suffix not in ['.pyc', '.pyo']:
            yield path


def check_private_material(path):
    relative = path.relative_to(ROOT)
    if path.suffix.lower() in {'.pem', '.key', '.safetensors', '.pt'} or path.name.startswith('.env'):
        raise RuntimeError('Excluded private/weight file type: ' + str(relative))
    if 'miniwob_html' in relative.parts:
        # Published, unmodified benchmark fixtures contain website paths/IP examples.
        # Verify their pinned identities instead of treating them as project secrets.
        return
    try:
        text = path.read_text()
    except UnicodeError:
        return
    if re.search(r'/(?:data\d+|Users|home|root)/', text):
        raise RuntimeError('Machine-specific absolute path: ' + str(relative))
    for match in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])', text):
        try:
            address = ipaddress.ip_address(match)
        except ValueError:
            continue
        if not address.is_loopback and not address.is_unspecified:
            raise RuntimeError('Unexpected IP address: ' + str(relative))
    # Public benchmark assets intentionally contain password tasks; inspect own code/docs.
    if 'miniwob_html' not in relative.parts:
        credential = r'''(?i)(?:password|passwd|api[_-]?key|secret)\s*[:=]\s*['"][^'"\n]{8,}['"]'''
        if re.search(credential, text):
            raise RuntimeError('Possible embedded credential: ' + str(relative))


def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    files = {str(p.relative_to(ROOT)): p for p in release_files()}
    if set(files) != set(manifest['files']):
        raise RuntimeError('Release file list differs from manifest')
    for name, path in files.items():
        check_private_material(path)
        entry = manifest['files'][name]
        if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Release file hash differs: ' + name)
    provenance = json.loads((ROOT / 'docs/source_provenance.json').read_text())
    for entry in provenance['files']:
        path = ROOT / entry['release_path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['release_sha256']:
            raise RuntimeError('Source provenance differs: ' + entry['release_path'])
    assets = ROOT / 'src/rsi_anything_gui/assets'
    for folder, record in [(assets / 'miniwob_html', 'task_inputs.json'),
                           (ROOT / 'vendor/jepa-anything-core/src/jepa_anything_core', 'core_inputs.json')]:
        expected = json.loads((assets / record).read_text())
        for name, sha in expected.items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Pinned upstream identity changed: ' + name)
    report = {'passed': True, 'files': len(files), 'byte_total': sum(p.stat().st_size for p in files.values()),
              'project_license_status': manifest['project_license_status'], 'full_gpu_rerun': False}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
