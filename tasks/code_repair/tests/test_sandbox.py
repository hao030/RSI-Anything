import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cure_wm


class SandboxTests(unittest.TestCase):
    def setUp(self):
        cure_wm._sandbox_ready.cache_clear()
        self.tests = (cure_wm.TestCase('', '1', 'one'),)

    def tearDown(self):
        cure_wm._sandbox_ready.cache_clear()

    def test_startup_failure_is_an_environment_error(self):
        failed = subprocess.CompletedProcess([], 1, '', 'runuser: may not be used by non-root users')
        with patch.object(cure_wm.shutil, 'which', side_effect=lambda name: '/usr/bin/' + name), \
             patch.object(cure_wm, '_run_probe', return_value=failed), \
             patch.object(cure_wm.subprocess, 'Popen') as candidate:
            with self.assertRaisesRegex(cure_wm.SandboxEnvironmentError, 'preflight failed'):
                cure_wm.execute('print(1)', self.tests)
            candidate.assert_not_called()

    def test_missing_launcher_is_an_environment_error(self):
        with patch.object(cure_wm.shutil, 'which', return_value=None):
            with self.assertRaises(cure_wm.SandboxEnvironmentError):
                cure_wm.execute('print(1)', self.tests)

    def test_preflight_timeout_is_an_environment_error(self):
        with patch.object(cure_wm.shutil, 'which', return_value='/usr/bin/launcher'), \
             patch.object(cure_wm, '_run_probe', side_effect=subprocess.TimeoutExpired('probe', 10)):
            with self.assertRaises(cure_wm.SandboxEnvironmentError):
                cure_wm.execute('print(1)', self.tests)

    def test_successful_preflight_is_reused_by_a_worker(self):
        ready = subprocess.CompletedProcess([], 0, 'RSI_SANDBOX_READY\n', '')
        process = Mock(returncode=0)
        process.communicate.return_value = ('1\n', '')
        with patch.object(cure_wm.shutil, 'which', return_value='/usr/bin/launcher'), \
             patch.object(cure_wm, '_run_probe', return_value=ready) as probe, \
             patch.object(cure_wm.subprocess, 'Popen', return_value=process):
            self.assertTrue(cure_wm.execute('print(1)', self.tests).all_pass)
            self.assertTrue(cure_wm.execute('print(1)', self.tests).all_pass)
            self.assertEqual(probe.call_count, 1)

    def test_candidate_runtime_error_remains_re_when_sandbox_works(self):
        ready = subprocess.CompletedProcess([], 0, 'RSI_SANDBOX_READY\n', '')
        process = Mock(returncode=1)
        process.communicate.return_value = ('', 'ZeroDivisionError')
        with patch.object(cure_wm.shutil, 'which', return_value='/usr/bin/launcher'), \
             patch.object(cure_wm, '_run_probe', return_value=ready) as probe, \
             patch.object(cure_wm.subprocess, 'Popen', return_value=process):
            result = cure_wm.execute('print(1 / 0)', self.tests)
            self.assertEqual(result.results[0].status, 'RE')
            self.assertEqual(probe.call_count, 2)

    def test_later_environment_failure_is_not_saved_as_re(self):
        ready = subprocess.CompletedProcess([], 0, 'RSI_SANDBOX_READY\n', '')
        failed = subprocess.CompletedProcess([], 1, '', 'unshare: Operation not permitted')
        process = Mock(returncode=1)
        process.communicate.return_value = ('', failed.stderr)
        with patch.object(cure_wm.shutil, 'which', return_value='/usr/bin/launcher'), \
             patch.object(cure_wm, '_run_probe', side_effect=[ready, failed]), \
             patch.object(cure_wm.subprocess, 'Popen', return_value=process):
            with self.assertRaises(cure_wm.SandboxEnvironmentError):
                cure_wm.execute('print(1)', self.tests)

    def test_local_execution_still_reports_ac_wa_ce_and_re(self):
        for code, status in [
            ('print(1)', 'AC'), ('print(2)', 'WA'), ('print(', 'CE'), ('raise ValueError()', 'RE'),
        ]:
            with self.subTest(status=status):
                result = cure_wm.execute(code, self.tests, sandbox='local_test')
                self.assertEqual(result.results[0].status, status)

    def test_probe_timeout_cleans_up_its_process_group(self):
        process = Mock(pid=2000, returncode=-9)
        process.communicate.side_effect = [subprocess.TimeoutExpired('probe', 10), ('', '')]
        with patch.object(cure_wm.subprocess, 'Popen', return_value=process), \
             patch.object(cure_wm.os, 'killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                cure_wm._run_probe(['sandbox-probe'])
            kill.assert_called_once_with(2000, cure_wm.signal.SIGKILL)
            self.assertEqual(process.communicate.call_count, 2)


if __name__ == '__main__':
    unittest.main()
