"""Small public command interface; heavy model imports remain lazy."""
import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='rsi-anything-gui')
    parser.add_argument('--config', type=Path, help='JSON configuration (relative paths resolve beside it)')
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor', help='Check environment, assets, model presence and optional GPUs')
    d.add_argument('--gpu', action='store_true')
    sub.add_parser('download-models', help='Download the two exact model revisions and verify every static file')
    sub.add_parser('verify-models', help='Verify the two local models against original static hashes')
    sub.add_parser('freeze', help='Hash base models and freeze the full experiment protocol')
    p = sub.add_parser('preflight', help='Run actual browser or GPU learning contracts')
    p.add_argument('phase', choices=['browser', 'world'])
    sub.add_parser('run', help='Execute the complete frozen protocol')
    sub.add_parser('audit', help='Audit a completed public-release run')
    sub.add_parser('aggregate', help='Recompute statistics for an audited run')
    s = sub.add_parser('smoke', help='Run actual browser rule episodes, no models or GPU required')
    s.add_argument('--episodes', type=int, default=4)
    args = parser.parse_args()
    if args.config:
        os.environ['RSI_GUI_CONFIG'] = str(args.config.expanduser().resolve())
    from . import settings
    if args.command == 'doctor':
        from .doctor import doctor
        result = doctor(args.gpu)
        print(json.dumps(result, indent=2))
        return 0 if result['passed'] else 1
    if args.command in ['download-models', 'verify-models']:
        from .models import prepare_models
        prepare_models(download=args.command == 'download-models')
        return 0
    if args.command == 'smoke':
        from .smoke import smoke
        print(json.dumps(smoke(args.episodes), indent=2))
        return 0
    if args.command == 'freeze':
        from .models import prepare_models
        prepare_models(download=False)
        from .protocol import bases, freeze
        bases()
        freeze()
        print('Frozen protocol: ' + str(settings.ROOT / 'protocol.json'))
        return 0
    settings.verify_frozen()
    if not (settings.ROOT / 'protocol.json').exists():
        parser.error('Run freeze before ' + args.command)
    if args.command == 'run' or (args.command == 'preflight' and args.phase == 'world'):
        from .models import prepare_models
        prepare_models(download=False)
    if args.command == 'preflight':
        env = dict(os.environ)
        if args.phase == 'world':
            gpu = settings.SETTINGS['resource_plan']['rsi_world_gpu']
            from .orchestrate import gpu_free
            gpu_free(gpu)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        return subprocess.call([sys.executable, '-m', 'rsi_anything_gui.contracts', args.phase], env=env)
    module = {'run': 'orchestrate', 'audit': 'audit', 'aggregate': 'aggregate'}[args.command]
    # Stay in the same process for the scheduler so its signal/finally cleanup executes.
    if module == 'orchestrate':
        importlib.import_module('.orchestrate', __package__).entrypoint()
        return 0
    importlib.import_module('.' + module, __package__).main()
    return 0
