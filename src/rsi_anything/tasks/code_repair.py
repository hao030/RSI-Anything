"""Code-repair recipe: data, four inherited world updates, then evaluation."""
import json
from ..workflow.types import Plan, Stage


def plan(config):
    options = config.options
    if set(options) - {'assets', 'gpus', 'train_gpu', 'inputs', 'checkpoint', 'judge_workers'}:
        raise ValueError('Unknown code-repair option')
    if config.mode not in {'full', 'evaluate', 'verify'}:
        raise ValueError('Code repair modes: full, evaluate, verify')
    if not options.get('assets'):
        raise ValueError('Code repair requires options.assets')
    if config.mode == 'full' and options.get('checkpoint'):
        raise ValueError('full uses the newly trained checkpoint; checkpoint belongs to evaluate mode')
    inputs = options.get('inputs', 'fresh' if config.mode == 'full' else 'recorded')
    if inputs not in {'fresh', 'recorded'}:
        raise ValueError('inputs must be fresh or recorded')
    gpus = options.get('gpus', '0,1,2,3,4,5,6,7')
    ids = gpus.split(',') if isinstance(gpus, str) else []
    if not ids or any(not d.isdigit() for d in ids) or len(set(ids)) != len(ids):
        raise ValueError('gpus must list unique nonnegative GPU IDs')
    train_gpu = options.get('train_gpu', 0)
    workers = options.get('judge_workers', 64)
    if type(train_gpu) is not int or train_gpu < 0 or type(workers) is not int or workers < 1:
        raise ValueError('Invalid train_gpu or judge_workers')
    backend = config.repository / 'tasks/code_repair'
    prefix = (config.python, str(backend / 'reproduce.py'))
    assets = ('--assets', options['assets'])
    stages = [Stage('verify-assets', prefix + ('verify-assets',) + assets, backend)]
    if config.mode == 'full':
        parent = None
        for name in ('initial', 'semantic', 'expanded', 'incumbent'):
            output = config.root / 'checkpoints/world' / name
            args = ('train',) + assets + ('--stage', name, '--output', str(output))
            if parent:
                args += ('--parent', str(parent))
            stages.append(Stage('update-world-' + name, prefix + args, backend,
                                {'CUDA_VISIBLE_DEVICES': str(train_gpu)},
                                (output / 'complete.json', output / 'joint_checkpoint'), (output,)))
            parent = output / 'joint_checkpoint'
        checkpoint = str(parent)
    else:
        checkpoint = options.get('checkpoint')
    if config.mode != 'verify':
        output = config.root / 'evaluations/code_repair'
        args = ('evaluate',) + assets + ('--output', str(output), '--inputs', inputs,
                    '--gpus', gpus, '--judge-workers', str(workers))
        if checkpoint:
            args += ('--checkpoint', checkpoint)
        stages.append(Stage('evaluate', prefix + args, backend,
                            expected=(output / 'summary.json', output / 'complete.json', output / 'results.jsonl'),
                            retry_outputs=(output,)))
    return Plan(tuple(stages))


def results(root):
    path = root / 'evaluations/code_repair/summary.json'
    return {'metrics': json.loads(path.read_text()) if path.exists() else {},
            'artifacts': {'evaluation': str(path)} if path.exists() else {}}
