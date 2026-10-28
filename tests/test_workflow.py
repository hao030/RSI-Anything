import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rsi_anything.tasks import adapter
from rsi_anything.workflow.config import RunConfig, load, source_identity
from rsi_anything.workflow.runner import run
from rsi_anything.workflow.types import Plan, Stage

REPOSITORY = Path(__file__).resolve().parents[1]


class WorkflowContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='rsi-workflow-test-')
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name).resolve()
        self.config = RunConfig('code_repair', 'verify', self.folder / 'run',
                                sys.executable, REPOSITORY, {'assets': str(self.folder / 'assets')})

    def stage(self, name, body, outputs=(), retry=()):
        return Stage(name, (sys.executable, '-c', body), self.folder,
                     expected=tuple(outputs), retry_outputs=tuple(retry))

    def execute_plan(self, plan, resume=False):
        with patch('rsi_anything.tasks.code_repair.plan', return_value=plan), contextlib.redirect_stdout(io.StringIO()):
            return run(self.config, resume)

    def test_real_stage_and_resume_skip_completed_work(self):
        output = self.config.root / 'proof.json'
        body = f'from pathlib import Path; p=Path({str(output)!r}); assert not p.exists(); p.write_text("ok")'
        plan = Plan((self.stage('collect', body, (output,)),))
        self.assertEqual(self.execute_plan(plan)['status'], 'completed')
        self.assertEqual(self.execute_plan(plan, True)['stages']['collect']['attempt'], 1)
        self.assertEqual(json.loads((self.config.root / 'summary.json').read_text())['status'], 'completed')

    def test_resume_detects_changed_stage_output(self):
        output = self.config.root / 'proof.json'
        plan = Plan((self.stage('collect', f'from pathlib import Path; Path({str(output)!r}).write_text("ok")', (output,)),))
        self.execute_plan(plan)
        output.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            self.execute_plan(plan, True)

    def test_failed_stage_stops_following_stages_and_records_log(self):
        marker = self.folder / 'unexpected'
        plan = Plan((self.stage('fail', 'print("failure detail"); raise SystemExit(7)'),
                     self.stage('next', f'from pathlib import Path; Path({str(marker)!r}).touch()')))
        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
            self.execute_plan(plan)
        state = json.loads((self.config.root / 'state.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertFalse(marker.exists())
        self.assertIn('failure detail', (self.config.root / state['stages']['fail']['log']).read_text())

    def test_retry_archives_partial_output_and_preserves_completed_stage(self):
        flag = self.folder / 'allow-success'
        output = self.config.root / 'checkpoints/one'
        body = (f'from pathlib import Path; p=Path({str(output)!r}); p.mkdir(parents=True); '
                f'(p/"result.json").write_text("ok"); raise SystemExit(0 if Path({str(flag)!r}).exists() else 3)')
        plan = Plan((self.stage('update', body, (output / 'result.json',), (output,)),))
        with self.assertRaises(RuntimeError):
            self.execute_plan(plan)
        flag.touch()
        result = self.execute_plan(plan, True)
        self.assertEqual(result['stages']['update']['attempt'], 2)
        self.assertNotIn('error', json.loads((self.config.root / 'state.json').read_text()))
        self.assertTrue((self.config.root / 'attempts/update/1/result.json').is_file())

    def test_success_exit_requires_expected_artifact(self):
        with self.assertRaisesRegex(RuntimeError, 'did not produce'):
            self.execute_plan(Plan((self.stage('empty', 'pass', (self.config.root / 'missing',)),)))

    def test_same_run_cannot_be_overwritten(self):
        plan = Plan((self.stage('one', 'pass'),))
        self.execute_plan(plan)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.execute_plan(plan)

    def test_changed_recipe_refuses_resume(self):
        self.execute_plan(Plan((self.stage('one', 'pass'),)))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.execute_plan(Plan((self.stage('two', 'pass'),)), True)

    def test_run_lock_refuses_second_owner(self):
        self.config.root.mkdir()
        with (self.config.root / '.workflow.lock').open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'Another workflow'):
                self.execute_plan(Plan(()))

    def test_foreign_directory_is_preserved(self):
        self.config.root.mkdir()
        marker = self.config.root / 'existing-result'
        marker.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'another run'):
            self.execute_plan(Plan(()))
        self.assertEqual(marker.read_text(), 'keep')

    def test_gui_runtime_namespace_and_paths(self):
        config = load(REPOSITORY / 'configs/gui/smoke.json')
        plan = adapter('gui').plan(config)
        self.assertIn('rsi_anything_gui', plan.stages[0].command)
        self.assertEqual(plan.files['task-config.json']['run_root'], str(config.root))
        self.assertIn('tasks/gui/src', plan.stages[0].env['PYTHONPATH'])

    def test_code_repair_checkpoint_chain_uses_this_run(self):
        config = load(REPOSITORY / 'configs/code_repair/default.json')
        stages = adapter('code_repair').plan(config).stages
        self.assertEqual(len(stages), 6)
        for i, name in enumerate(['initial', 'semantic', 'expanded', 'incumbent'], 1):
            self.assertEqual(stages[i].name, 'update-world-' + name)
            if i > 1:
                parent = stages[i].command[stages[i].command.index('--parent') + 1]
                self.assertEqual(parent, str(config.root / 'checkpoints/world' / ['initial', 'semantic', 'expanded'][i-2] / 'joint_checkpoint'))
        final = stages[-1].command
        self.assertEqual(final[final.index('--checkpoint') + 1], str(config.root / 'checkpoints/world/incumbent/joint_checkpoint'))

    def test_relative_settings_and_venv_symlink_preserved(self):
        interpreter = self.folder / 'venv/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        recipe = self.folder / 'recipe.json'
        recipe.write_text(json.dumps({'task': 'gui', 'mode': 'smoke', 'repository': str(REPOSITORY),
                                     'run_root': 'outputs', 'python': 'venv/bin/python'}))
        config = load(recipe)
        self.assertEqual(config.python, str(interpreter))
        self.assertEqual(config.root, self.folder / 'outputs')

    def test_plan_runs_without_ml_imports_or_run_directory(self):
        recipe = self.folder / 'recipe.json'
        recipe.write_text(json.dumps({'task': 'gui', 'mode': 'smoke', 'repository': str(REPOSITORY), 'run_root': 'new-run'}))
        env = dict(os.environ, PYTHONPATH=str(REPOSITORY / 'src'))
        result = subprocess.run([sys.executable, '-m', 'rsi_anything', 'plan', '--config', str(recipe)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.folder / 'new-run').exists())
        self.assertEqual(json.loads(result.stdout)['config']['task'], 'gui')

    def test_resume_detects_changed_checkpoint_contents(self):
        checkpoint = self.config.root / 'checkpoint'
        body = (f'from pathlib import Path; p=Path({str(checkpoint)!r}); '
                'p.mkdir(); (p/"weights.bin").write_bytes(b"original")')
        plan = Plan((self.stage('train', body, (checkpoint,)),))
        self.execute_plan(plan)
        (checkpoint / 'weights.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            self.execute_plan(plan, True)

    def test_source_identity_includes_browser_assets(self):
        from dataclasses import replace
        repository = self.folder / 'source'
        assets = repository / 'tasks/gui/src/rsi_anything_gui/assets'
        assets.mkdir(parents=True)
        page = assets / 'task.html'
        page.write_text('<button>Before</button>')
        config = replace(self.config, repository=repository, task='gui')
        before = source_identity(config)
        page.write_text('<button>After</button>')
        self.assertNotEqual(source_identity(config), before)

    def test_invalid_recipe_types_have_actionable_cli_errors(self):
        from rsi_anything.cli import main
        for field, value in [('task', []), ('mode', []), ('python', None),
                             ('repository', 1), ('options', {'assets': 7})]:
            with self.subTest(field=field):
                recipe = self.folder / 'invalid.json'
                data = {'task': 'gui', 'mode': 'smoke', 'repository': str(REPOSITORY),
                        'run_root': 'output'}
                data[field] = value
                recipe.write_text(json.dumps(data))
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(['plan', '--config', str(recipe)]), 1)
                self.assertTrue(stderr.getvalue().strip())
                self.assertNotIn('Traceback', stderr.getvalue())

    def test_source_directory_cannot_be_run_root(self):
        recipe = self.folder / 'recipe.json'
        recipe.write_text(json.dumps({'task': 'gui', 'repository': str(REPOSITORY), 'run_root': str(REPOSITORY / 'tasks/gui')}))
        with self.assertRaisesRegex(ValueError, 'separate from source'):
            load(recipe)


if __name__ == '__main__':
    unittest.main()
