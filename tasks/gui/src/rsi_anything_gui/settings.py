"""Portable runtime settings and immutable code/input identities."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
MINIWoB_COMMIT = (PACKAGE / 'assets/miniwob_commit.txt').read_text().strip()
DEFAULT_RESOURCES = {'policy_gpu': 0, 'workers': [1], 'rsi_world_gpu': 1, 'rsi_policy_gpu': 2}


def validate_resources(plan):
    if set(plan) != set(DEFAULT_RESOURCES):
        raise ValueError('Resource plan requires policy_gpu, workers, rsi_world_gpu, rsi_policy_gpu')
    workers = plan['workers']
    if not isinstance(workers, list) or not workers or len(workers) != len(set(workers)):
        raise ValueError('workers must be a nonempty list of unique physical GPU IDs')
    ids = workers + [plan[k] for k in ['policy_gpu', 'rsi_world_gpu', 'rsi_policy_gpu']]
    if any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('GPU IDs must be nonnegative integers')
    if plan['policy_gpu'] in workers or plan['policy_gpu'] in [plan['rsi_world_gpu'], plan['rsi_policy_gpu']]:
        raise ValueError('The persistent policy service requires a separate GPU')
    if plan['rsi_world_gpu'] == plan['rsi_policy_gpu']:
        raise ValueError('RSI world and policy training require separate GPUs')


def load_settings():
    config = os.environ.get('RSI_GUI_CONFIG')
    path = Path(config).expanduser().resolve() if config else None
    data = json.loads(path.read_text()) if path else {}
    allowed = {'run_root', 'world_model', 'policy_model', 'assets', 'policy_port', 'resource_plan'}
    if set(data) - allowed:
        raise ValueError('Unknown settings: ' + ', '.join(sorted(set(data) - allowed)))
    base = path.parent if path else Path.cwd()

    def resolve(value):
        p = Path(value).expanduser()
        return (p if p.is_absolute() else base / p).resolve()

    run = resolve(os.environ.get('RSI_GUI_ROOT') or data.get('run_root', 'runs/full'))
    world = resolve(os.environ.get('RSI_GUI_WORLD_MODEL') or data.get('world_model') or run / 'models/Qwen2.5-Coder-3B-Instruct')
    policy = resolve(os.environ.get('RSI_GUI_POLICY_MODEL') or data.get('policy_model') or run / 'models/ReasonFlux-Coder-7B')
    assets = resolve(os.environ.get('RSI_GUI_ASSETS') or data.get('assets') or PACKAGE / 'assets/miniwob_html')
    port = int(os.environ.get('RSI_GUI_POLICY_PORT') or data.get('policy_port', 8767))
    if not 1024 <= port <= 65535:
        raise ValueError('policy_port must be between 1024 and 65535')
    plan = data.get('resource_plan', dict(DEFAULT_RESOURCES, workers=[1]))
    validate_resources(plan)
    if world.name != 'Qwen2.5-Coder-3B-Instruct' or policy.name != 'ReasonFlux-Coder-7B':
        raise ValueError('Model directory basenames must match the pinned model names')
    return dict(run_root=run, world_model=world, policy_model=policy, assets=assets, policy_port=port, resource_plan=plan)


SETTINGS = load_settings()
ROOT = SETTINGS['run_root']
BASE_WORLD = SETTINGS['world_model']
BASE_POLICY = SETTINGS['policy_model']
ASSETS = SETTINGS['assets']
PORT = SETTINGS['policy_port']


def runtime_identity():
    result = {k: str(v) if isinstance(v, Path) else v for k, v in SETTINGS.items()}
    result['playwright_browsers_path'] = os.environ.get('PLAYWRIGHT_BROWSERS_PATH')
    return result


def _hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    # Covers actual runtime source and task assets, including the vendored core.
    spec = importlib.util.find_spec('jepa_anything_core')
    if spec is None or spec.origin is None:
        raise RuntimeError('Install the release before freezing its protocol')
    core = Path(spec.origin).parent
    result = {'rsi_anything_gui/' + p.name: _hash(p) for p in sorted(PACKAGE.glob('*.py'))}
    core_hashes = {p.name: _hash(p) for p in sorted(core.glob('*.py'))}
    expected_core = json.loads((PACKAGE / 'assets/core_inputs.json').read_text())
    if core_hashes != expected_core:
        raise RuntimeError('JEPA core differs from the validated bundled source; reinstall this release')
    result.update({'jepa_anything_core/' + name: sha for name, sha in core_hashes.items()})
    asset_hashes = {str(p.relative_to(ASSETS)): _hash(p) for p in sorted(ASSETS.rglob('*')) if p.is_file()}
    expected_assets = json.loads((PACKAGE / 'assets/task_inputs.json').read_text())
    if asset_hashes != expected_assets:
        raise RuntimeError('Official task assets differ from the pinned snapshot')
    result.update({'assets/' + name: sha for name, sha in asset_hashes.items()})
    for p in sorted((PACKAGE / 'assets').glob('*')):
        if p.is_file():
            result['metadata/' + p.name] = _hash(p)
    return result


def verify_frozen():
    path = ROOT / 'protocol.json'
    if not path.exists():
        return
    p = json.loads(path.read_text())
    if p.get('runtime_identity') != runtime_identity() or p.get('source_hashes') != source_hashes():
        raise RuntimeError('Frozen runtime/source changed; restore it or use a new run directory')
    manifest = ROOT / 'base_manifest.json'
    if not manifest.exists():
        raise RuntimeError('Frozen base manifest is missing')
    identity = hashlib.sha256(json.dumps(json.loads(manifest.read_text()), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if identity != p['base_manifest_hash']:
        raise RuntimeError('Frozen base manifest changed')
