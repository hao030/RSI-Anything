"""GUI recipe using an isolated worker namespace and native task scheduler."""
import json
import os
from ..workflow.types import Plan, Stage


def plan(config):
    options = dict(config.options)
    if set(options) - {'world_model', 'policy_model', 'assets', 'policy_port', 'resource_plan', 'episodes', 'download_models'}:
        raise ValueError('Unknown GUI option')
    if config.mode not in {'full', 'smoke'}:
        raise ValueError('GUI modes: full, smoke')
    episodes = options.pop('episodes', 4)
    download = options.pop('download_models', True)
    if type(episodes) is not int or not 1 <= episodes <= 100 or type(download) is not bool:
        raise ValueError('episodes must be 1..100; download_models must be boolean')
    backend = config.repository / 'tasks/gui'
    runtime = dict(options, run_root=str(config.root))
    settings = config.root / 'task-config.json'
    prefix = (config.python, '-m', 'rsi_anything_gui', '--config', str(settings))
    env = {'PYTHONPATH': os.pathsep.join([str(backend / 'src'), str(backend / 'vendor/jepa-anything-core/src')])}

    def stage(name, args, expected=()):
        return Stage(name, prefix + args, backend, env, tuple(config.root / p for p in expected))

    if config.mode == 'smoke':
        stages = [stage('browser-smoke', ('smoke', '--episodes', str(episodes)), ('results/browser_smoke.json',))]
    else:
        stages = []
        if download:
            stages.append(stage('prepare-models', ('download-models',)))
        stages.extend([
            stage('check-environment', ('doctor', '--gpu')),
            stage('freeze-protocol', ('freeze',), ('protocol.json', 'base_manifest.json')),
            stage('preflight-browser', ('preflight', 'browser'), ('results/browser_contracts.json',)),
            stage('preflight-world', ('preflight', 'world'), ('results/world_contracts.json',)),
            stage('adapt-and-evaluate', ('run',), ('outputs/results.json', 'outputs/completion_audit.json')),
        ])
    return Plan(tuple(stages), {'task-config.json': runtime})


def results(root):
    path = root / 'outputs/results.json'
    if path.exists():
        result = json.loads(path.read_text())
        return {'metrics': {'arms': result['arms']}, 'artifacts': {'evaluation': str(path)}}
    path = root / 'results/browser_smoke.json'
    if path.exists():
        result = json.loads(path.read_text())
        episodes = result['episodes']
        return {'metrics': {'episodes': len(episodes), 'successes': sum(r['success'] for r in episodes),
                            'passed': result['passed']}, 'artifacts': {'browser_smoke': str(path)}}
    return {'metrics': {}, 'artifacts': {}}
