"""Resolve a task recipe without importing either task's ML environment."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Paths must be nonempty strings')
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class RunConfig:
    task: str
    mode: str
    root: Path
    python: str
    repository: Path
    options: dict

    def as_dict(self):
        return dict(task=self.task, mode=self.mode, run_root=str(self.root),
                    python=self.python, repository=str(self.repository), options=self.options)


def load(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) - {'task', 'mode', 'run_root', 'python', 'repository', 'options'}:
        raise ValueError('Recipe requires task, run_root and optional mode, python, repository, options')
    if not isinstance(data.get('task'), str) or data['task'] not in {'code_repair', 'gui'}:
        raise ValueError('task must be code_repair or gui')
    if not isinstance(data.get('run_root'), str) or not data['run_root']:
        raise ValueError('run_root must be a nonempty path')
    if not isinstance(data.get('mode', 'full'), str):
        raise ValueError('mode must be a string')
    repository = resolve(path.parent, data['repository']) if 'repository' in data else Path(__file__).resolve().parents[3]
    if not (repository / 'tasks' / data['task']).is_dir():
        raise ValueError('Use a source checkout or set repository to its path')
    root = resolve(path.parent, data['run_root'])
    # Runs may live under runs/; source/config/task directories are never run roots.
    protected = [repository / name for name in ['src', 'tasks', 'configs', 'docs', 'scripts', 'tests', '.git']]
    if root == repository or repository.is_relative_to(root) or any(root.is_relative_to(p) for p in protected):
        raise ValueError('run_root must be separate from source and configuration directories')
    interpreter = data.get('python', 'current')
    if not isinstance(interpreter, str) or not interpreter.strip():
        raise ValueError('python must be an executable name or path')
    if interpreter == 'current':
        interpreter = sys.executable
    elif '/' in interpreter or '\\' in interpreter:
        # Keep a venv executable's symlink path: resolving it loses the venv.
        interpreter = str(Path(interpreter).expanduser()) if Path(interpreter).expanduser().is_absolute() else str(path.parent / interpreter)
        interpreter = os.path.abspath(interpreter)
    else:
        interpreter = shutil.which(interpreter) or interpreter
    options = data.get('options', {})
    if not isinstance(options, dict):
        raise ValueError('options must be an object')
    options = dict(options)
    for name in ('assets', 'checkpoint', 'world_model', 'policy_model'):
        if options.get(name) is not None:
            options[name] = str(resolve(path.parent, options[name]))
    return RunConfig(data['task'], data.get('mode', 'full'), root, interpreter, repository, options)


def source_identity(config):
    backend = config.repository / 'tasks' / config.task
    folders = [config.repository / 'src'] + [backend / name for name in ('src', 'vendor', 'config', 'configs')]
    skipped = {'__pycache__', '.venv', 'runs', 'build', 'dist', '.pytest_cache'}
    files = {}
    for path in sorted(backend.glob('*')):
        if path.is_file() and path.suffix in {'.py', '.toml', '.txt'}:
            files[str(path.relative_to(config.repository))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            relative = path.relative_to(folder)
            if any(p in skipped or p.endswith('.egg-info') for p in relative.parts):
                continue
            # Browser HTML, scripts, styles and media also define task behavior.
            if path.is_file():
                files[str(path.relative_to(config.repository))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(files)
