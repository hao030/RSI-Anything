"""Opt-in actual Chromium validation; no base model or GPU is loaded."""
import json
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('RSI_TEST_BROWSER') != '1', reason='set RSI_TEST_BROWSER=1 after installing Chromium')


def test_actual_browser_contracts_and_interrupted_recovery():
    from rsi_anything_gui.common import ROOT, atomic
    from rsi_anything_gui.protocol import freeze
    from rsi_anything_gui.contracts import browser
    # Unit/browser fixture only; the public freeze command verifies real model inputs.
    atomic(ROOT / 'base_manifest.json', {'browser_test_stub': {}})
    freeze()
    browser()
    report = json.loads((ROOT / 'results/browser_contracts.json').read_text())
    assert report['passed'] and len(report['checks']) >= 13
    assert report['checks']['resume_skips_fully_committed_episode']
    assert report['checks']['failed_episode_positive_feedback_rejected']
