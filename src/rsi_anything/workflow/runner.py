"""One stage runner for both tasks, with exclusive ownership and resumable records."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .config import digest, source_identity
from ..tasks import adapter


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def output_hash(path):
    """Record complete checkpoint directories as well as individual result files."""
    path = Path(path)
    if path.is_dir():
        return digest({str(p.relative_to(path)): file_hash(p)
                       for p in sorted(path.rglob('*')) if p.is_file()})
    return file_hash(path)


def preview(config):
    plan = adapter(config.task).plan(config)
    return {'config': config.as_dict(), 'stages': [s.as_dict() for s in plan.stages], 'files': plan.files}


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Descendants may survive an already-exited group leader.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(stage, log):
    env = {k: v for k, v in os.environ.items() if not k.startswith('RSI_GUI_')}
    env.update(stage.env)
    env['PYTHONUNBUFFERED'] = '1'
    with log.open('w') as stream:
        process = subprocess.Popen(stage.command, cwd=stage.cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
            if code:
                raise RuntimeError(f'{stage.name} exited {code}; see {log}')
        except BaseException:
            _stop(process)
            raise


def report(root):
    root = Path(root).resolve()
    state = json.loads((root / 'state.json').read_text())
    result = adapter(state['task']).results(root)
    return dict(task=state['task'], mode=state['mode'], status=state['status'],
                run_root=str(root), stages=state['stages'], **result)


def run(config, resume=False):
    implementation = adapter(config.task)
    plan = implementation.plan(config)
    identity = digest({'config': config.as_dict(), 'source': source_identity(config),
                       'plan': [s.as_dict() for s in plan.stages], 'files': plan.files})
    config.root.mkdir(parents=True, exist_ok=True)
    with (config.root / '.workflow.lock').open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another workflow owns this run directory') from error
        state_path = config.root / 'state.json'
        if state_path.exists():
            if not resume:
                raise ValueError('Run already exists; use --resume or choose a new run_root')
            state = json.loads(state_path.read_text())
            if state['identity'] != identity:
                raise ValueError('Configuration or source changed; choose a new run_root')
            if json.loads((config.root / 'config.json').read_text()) != config.as_dict():
                raise ValueError('Saved workflow configuration changed')
            for name, content in plan.files.items():
                if json.loads((config.root / name).read_text()) != content:
                    raise ValueError('Task configuration changed: ' + name)
        else:
            if resume:
                raise ValueError('No saved workflow to resume')
            if any(p.name != '.workflow.lock' for p in config.root.iterdir()):
                raise ValueError('run_root contains files from another run; choose an empty directory')
            state = dict(schema_version=1, identity=identity, task=config.task, mode=config.mode,
                         status='pending', stages={}, started_at=time.time())
            atomic(config.root / 'config.json', config.as_dict())
            for name, content in plan.files.items():
                atomic(config.root / name, content)
            atomic(state_path, state)

        # Verify completed stage receipts before starting any further work.
        for record in state['stages'].values():
            if record['status'] == 'completed':
                for relative, expected in record['outputs'].items():
                    path = config.root / relative
                    if not path.exists() or output_hash(path) != expected:
                        raise ValueError('Completed stage output changed: ' + relative)

        def interrupted(signum, frame):
            raise KeyboardInterrupt('Workflow interrupted')

        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        stage = None
        try:
            state.pop('error', None)
            state.pop('finished_at', None)
            state['status'] = 'running'
            atomic(state_path, state)
            for stage in plan.stages:
                old = state['stages'].get(stage.name, {})
                if old.get('status') == 'completed':
                    continue
                attempt = old.get('attempt', 0) + 1
                # Preserve incomplete outputs from tasks that require a fresh output path.
                for path in stage.retry_outputs:
                    if path.exists():
                        if not path.is_relative_to(config.root) or not old:
                            raise ValueError('Unexpected stage output: ' + str(path))
                        archive = config.root / 'attempts' / stage.name / str(attempt - 1)
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        path.rename(archive)
                log = config.root / 'logs/workflow' / f'{stage.name}.{attempt}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                record = dict(status='running', attempt=attempt, started_at=time.time(),
                              log=str(log.relative_to(config.root)), outputs={})
                state['stages'][stage.name] = record
                state['current_stage'] = stage.name
                atomic(state_path, state)
                print(f'[{stage.name}] running → {log}', flush=True)
                execute(stage, log)
                for path in stage.expected:
                    if not path.is_file() and not path.is_dir():
                        raise RuntimeError(f'{stage.name} did not produce {path}')
                    record['outputs'][str(path.relative_to(config.root))] = output_hash(path)
                record.update(status='completed', finished_at=time.time())
                atomic(state_path, state)
                print(f'[{stage.name}] completed', flush=True)
            state.update(status='completed', finished_at=time.time(), current_stage=None)
            atomic(state_path, state)
            summary = report(config.root)
            atomic(config.root / 'summary.json', summary)
            return summary
        except BaseException as error:
            status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            state.update(status=status, error=str(error), updated_at=time.time())
            if stage and state['stages'].get(stage.name, {}).get('status') == 'running':
                state['stages'][stage.name].update(status=status, error=str(error))
            atomic(state_path, state)
            # Keep summary status current after a failed or interrupted resume.
            atomic(config.root / 'summary.json', {'task': config.task, 'mode': config.mode,
                   'status': status, 'stages': state['stages'], 'metrics': {}, 'error': str(error)})
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
