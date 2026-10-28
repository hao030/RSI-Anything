"""Shared source-checkout interface for RSI task workflows."""
import argparse
import json
from pathlib import Path
import sys

from .workflow.config import load
from .workflow.runner import preview, report, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rsi', description='Run an RSI workflow with code-repair or GUI task adapters')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('tasks', help='List available tasks and recipes')
    assets = commands.add_parser('assets', help='Download or verify versioned experiment assets')
    assets.add_argument('task', choices=['code_repair'])
    assets.add_argument('--output', type=Path, default=Path('assets'))
    assets.add_argument('--profile', choices=['evaluate', 'full'], default='evaluate')
    assets.add_argument('--skip-models', action='store_true', help='Install data/checkpoints only')
    assets.add_argument('--verify', action='store_true', help='Verify selected release files without downloading')
    for name in ('plan', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True, type=Path)
        if name == 'run':
            command.add_argument('--resume', action='store_true')
    for name in ('status', 'report'):
        commands.add_parser(name).add_argument('--run', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'tasks':
            result = {'code_repair': ['full', 'evaluate', 'verify'], 'gui': ['full', 'smoke']}
        elif args.command == 'assets':
            from .assets import install
            result = install(args.output, args.profile, models=not args.skip_models, verify_only=args.verify)
        elif args.command == 'plan':
            result = preview(load(args.config))
        elif args.command == 'run':
            result = run(load(args.config), args.resume)
        elif args.command == 'status':
            result = json.loads((args.run / 'state.json').read_text())
        else:
            result = report(args.run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print('Interrupted; run state and stage logs saved.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
