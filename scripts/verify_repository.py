"""Verify the reviewed repository manifest and its two experiment boundaries."""
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_files():
    output = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z'])
    return {p.decode() for p in output.split(b'\0') if p and p.decode() != 'SHA256SUMS.json'}


def main():
    manifest = json.loads((ROOT / 'SHA256SUMS.json').read_text())
    if set(manifest) != tracked_files():
        raise RuntimeError('Tracked repository files differ from SHA256SUMS.json')
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Repository file identity changed: ' + relative)
        # Public benchmark fixtures retain their upstream examples and exact bytes.
        if 'miniwob_html' in path.parts or path.suffix in {'.png', '.pdf'}:
            continue
        try:
            text = path.read_text()
        except UnicodeError:
            continue
        if re.search(r'/(?:data\d+|Users|home|root)/', text):
            raise RuntimeError('Machine-specific absolute path: ' + relative)
        if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}', text):
            raise RuntimeError('Possible embedded credential: ' + relative)
    gui = ROOT / 'tasks/gui'
    core = 'vendor/jepa-anything-core/src/jepa_anything_core'
    a = {p.name: sha(p) for p in (ROOT / 'tasks/code_repair' / core).glob('*.py')}
    b = {p.name: sha(p) for p in (gui / core).glob('*.py')}
    if a != b or len(a) != 5:
        raise RuntimeError('Code-repair and GUI JEPA core identities differ')
    required = gui / 'src/rsi_anything_gui/assets/miniwob_html/miniwob/click-checkboxes.html'
    if str(required.relative_to(ROOT)) not in manifest:
        raise RuntimeError('GUI official task HTML is not tracked')
    spec = importlib.util.spec_from_file_location('gui_release_verifier', gui / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    print(json.dumps({'passed': True, 'repository_files': len(manifest),
                      'identical_jepa_core_files': len(a), 'gui_task_assets_tracked': True}))


if __name__ == '__main__':
    main()
