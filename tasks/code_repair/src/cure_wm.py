from __future__ import annotations
import os, resource, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from functools import lru_cache


class SandboxEnvironmentError(RuntimeError):
    """The judging environment could not start a known correct program."""


@contextmanager
def _program_script(code: str):
    with tempfile.TemporaryDirectory(prefix='cure_wm_') as tmp:
        tmp_dir = Path(tmp)
        tmp_dir.chmod(0o755)
        script = tmp_dir / 'candidate.py'
        script.write_text(code, encoding='utf-8')
        script.chmod(0o644)
        yield script


def _sandbox_tools():
    runuser, unshare = shutil.which('runuser'), shutil.which('unshare')
    if not runuser or not unshare:
        raise SandboxEnvironmentError('Required runuser/unshare sandbox is unavailable')
    return runuser, unshare


def _sandbox_command(script, tools):
    runuser, unshare = tools
    return [runuser, '-u', 'nobody', '--', unshare, '-Urn', '--', sys.executable, '-I', str(script)]


def _execution_env():
    return {'PATH': '/usr/bin:/bin', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}


def _run_probe(command):
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, preexec_fn=_limits, env=_execution_env(),
    )
    try:
        stdout, stderr = proc.communicate('', timeout=10.)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _probe_sandbox(tools):
    marker = 'RSI_SANDBOX_READY'
    with _program_script(f'print({marker!r})') as script:
        try:
            result = _run_probe(_sandbox_command(script, tools))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxEnvironmentError(f'Sandbox preflight could not start Python: {exc}') from exc
    if result.returncode or result.stdout.strip() != marker:
        detail = result.stderr.strip()[-2048:] or f'exit {result.returncode}, unexpected output'
        raise SandboxEnvironmentError(f'Sandbox preflight failed: {detail}')


@lru_cache(maxsize=8)
def _sandbox_ready(tools, executable, user_id, process_id):
    _probe_sandbox(tools)


def ensure_sandbox():
    """Validate once per worker; failures are not cached as candidate results."""
    tools = _sandbox_tools()
    _sandbox_ready(tools, sys.executable, os.geteuid(), os.getpid())

@dataclass(frozen=True)
class TestCase:
    input: str
    expected: str
    test_id: str

@dataclass(frozen=True)
class TestResult:
    test_id: str
    status: str
    actual: str
    expected: str
    stderr: str
    seconds: float

@dataclass(frozen=True)
class Execution:
    results: tuple[TestResult, ...]
    wall_seconds: float

    @property
    def all_pass(self) -> bool:
        return bool(self.results) and all((r.status == 'AC' for r in self.results))

    @property
    def pass_fraction(self) -> float:
        return sum((r.status == 'AC' for r in self.results)) / len(self.results) if self.results else 0.0

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    code: str

def _limits():
    if sys.platform != 'linux':
        return
    resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 ** 2, 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

def _run_one(script: Path, test: TestCase, timeout: float, sandbox: str) -> TestResult:
    started = time.monotonic()
    if sandbox == 'required':
        cmd = _sandbox_command(script, _sandbox_tools())
    elif sandbox == 'local_test':
        cmd = [sys.executable, '-I', str(script)]
    else:
        raise ValueError(f'unknown sandbox: {sandbox}')
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=_limits, env=_execution_env())
    except OSError as exc:
        if sandbox == 'required':
            raise SandboxEnvironmentError(f'Sandbox launcher failed: {exc}') from exc
        raise
    try:
        stdout, stderr = proc.communicate(test.input, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return TestResult(test.test_id, 'TLE', '', test.expected, 'timeout', time.monotonic() - started)
    elapsed = time.monotonic() - started
    if proc.returncode:
        if sandbox == 'required':
            # Recheck before calling this a candidate error: the environment may
            # have changed since the worker's initial successful preflight.
            _probe_sandbox(_sandbox_tools())
        return TestResult(test.test_id, 'RE', stdout[-1024:], test.expected, stderr[-2048:], elapsed)
    status = 'AC' if stdout.strip() == test.expected.strip() else 'WA'
    return TestResult(test.test_id, status, stdout[-1024:], test.expected, stderr[-2048:], elapsed)

def execute(code: str, tests: tuple[TestCase, ...], *, timeout: float=3.0, sandbox: str='required') -> Execution:
    if not tests:
        raise ValueError('Cannot execute an empty test suite')
    if not code.strip():
        started = time.monotonic()
        results = tuple((TestResult(t.test_id, 'CE', '', t.expected, 'Empty candidate program', 0.0) for t in tests))
        return Execution(results, time.monotonic() - started)
    try:
        compile(code, 'candidate.py', 'exec')
    except SyntaxError as exc:
        started = time.monotonic()
        msg = f'{type(exc).__name__}: {exc.msg} (line {exc.lineno})'
        results = tuple((TestResult(t.test_id, 'CE', '', t.expected, msg, 0.0) for t in tests))
        return Execution(results, time.monotonic() - started)
    if sandbox == 'required':
        ensure_sandbox()
    started = time.monotonic()
    with _program_script(code) as script:
        results = tuple((_run_one(script, test, timeout, sandbox) for test in tests))
    return Execution(results, time.monotonic() - started)
