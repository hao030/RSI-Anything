import collections
import json
from unittest.mock import Mock

import pytest

from rsi_anything_gui import experiment
from rsi_anything_gui.common import legal_actions
from rsi_anything_gui.gui_env import GUIEnv, rule_action


def proposal(obs):
    actions = legal_actions(obs)
    best = actions.index(rule_action(obs))
    order = [best] + [i for i in range(len(actions)) if i != best]
    pool = order[:4]
    if len(actions) - 1 not in pool:
        pool = pool[:3] + [len(actions) - 1]
    probabilities = [.3 / (len(actions) - 1)] * len(actions)
    probabilities[best] = .7
    return {
        'all_action_ids': order, 'candidate_ids': pool,
        'actions': [actions[i] for i in pool], 'probabilities': probabilities,
        'seconds': .01, 'cached': False,
    }


@pytest.fixture
def browser_env():
    import os
    if os.environ.get('RSI_TEST_BROWSER') != '1':
        pytest.skip('set RSI_TEST_BROWSER=1 after installing Chromium')
    env = GUIEnv()
    try:
        yield env
    finally:
        env.close()


def test_timeout_during_policy_becomes_a_zero_action_failure(browser_env, monkeypatch):
    def slow_rank(obs, adapter=None):
        browser_env.started -= 121
        return proposal(obs)
    monkeypatch.setattr(experiment, 'rank', slow_rank)
    result, trace = experiment.episode(browser_env, 190000)
    assert result['success'] == 0 and result['steps'] == 0 and trace == []
    assert result['termination']['reason'] == 'timeout'
    assert result['final']['done'] and result['final']['budget_exhausted']
    assert result['initial']['elements'] == result['final']['elements']


def test_world_forecast_is_skipped_after_policy_timeout(browser_env, monkeypatch):
    def slow_rank(obs, adapter=None):
        browser_env.started -= 121
        return proposal(obs)
    monkeypatch.setattr(experiment, 'rank', slow_rank)
    model = Mock()
    result, trace = experiment.episode(browser_env, 190000, 'world', model=model)
    assert result['success'] == 0 and trace == []
    model.forecast.assert_not_called()


def test_timeout_during_world_forecast_is_recorded(browser_env, monkeypatch):
    monkeypatch.setattr(experiment, 'rank', lambda obs, adapter=None: proposal(obs))
    def slow_forecast(contexts):
        browser_env.started -= 121
        return [{'value': 0., 'submit_success_logit': 0.} for _ in contexts], .1
    model = Mock()
    model.forecast.side_effect = slow_forecast
    result, trace = experiment.episode(browser_env, 190000, 'world', model=model)
    assert result['success'] == 0 and result['steps'] == 0 and trace == []
    assert result['termination']['reason'] == 'timeout'
    assert result['world_seconds'] == .1


@pytest.mark.parametrize('timeout_step', [0, 1])
def test_timeout_rollout_continues_audits_and_resumes(tmp_path, monkeypatch, timeout_step):
    import os
    if os.environ.get('RSI_TEST_BROWSER') != '1':
        pytest.skip('set RSI_TEST_BROWSER=1 after installing Chromium')
    from rsi_anything_gui import audit
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    (tmp_path / 'protocol.json').write_text(json.dumps({'test': 'timeout recovery'}))
    env = GUIEnv()
    monkeypatch.setattr(experiment, 'GUIEnv', lambda: env)
    def rank(obs, adapter=None):
        if env.seed == 190000 and env.steps == timeout_step:
            env.started -= 121
        return proposal(obs)
    reset = env.reset
    def reset_seed(seed):
        env.seed = seed
        return reset(seed)
    monkeypatch.setattr(env, 'reset', reset_seed)
    monkeypatch.setattr(experiment, 'rank', rank)
    path = tmp_path / 'episodes.jsonl'
    results = experiment.rollout([190000, 190001], path)
    assert [r['success'] for r in results] == [0, 1]
    assert results[0]['steps'] == timeout_step
    monkeypatch.setattr(audit, 'errors', [])
    monkeypatch.setattr(audit, 'counts', collections.Counter())
    assert audit.episode_file(path, [190000, 190001])['success_rate'] == .5
    assert audit.errors == []
    # Fully committed failures are resumed just like successful episodes.
    env = GUIEnv()
    monkeypatch.setattr(experiment, 'GUIEnv', lambda: env)
    rank_mock = Mock(side_effect=AssertionError('committed episode must not be rerun'))
    monkeypatch.setattr(experiment, 'rank', rank_mock)
    assert experiment.rollout([190000, 190001], path) == results
    rank_mock.assert_not_called()


def test_infrastructure_errors_remain_errors(monkeypatch):
    env = Mock()
    env.reset.return_value = {
        'instruction': 'Select alpha and click Submit.',
        'elements': [{'slot': 0, 'label': 'alpha', 'checked': False}],
        'done': False, 'submitted': False, 'remaining_actions': 8,
    }
    monkeypatch.setattr(experiment, 'rank', Mock(side_effect=RuntimeError('policy service unavailable')))
    with pytest.raises(RuntimeError, match='policy service unavailable'):
        experiment.episode(env, 1)
