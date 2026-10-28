import json
import random
import subprocess
import sys
from pathlib import Path
import pytest

from rsi_anything_gui import settings
from rsi_anything_gui.common import context, target, correctness, label_transition, atomic


def observation():
    return {'instruction': 'Select alpha and click Submit.', 'elements': [
        {'slot': 0, 'ref': 'ch0', 'label': 'alpha', 'checked': False},
        {'slot': 1, 'ref': 'ch1', 'label': 'beta', 'checked': False}],
        'remaining_actions': 8, 'done': False, 'submitted': False,
        'raw_reward': -1, 'reward_reason': 'private evaluator field', 'target_bits': [True, False]}


def test_public_encoder_inputs_exclude_evaluator():
    obs = observation()
    for encoded in [json.loads(context(obs, {'kind': 'click', 'slot': 0})), json.loads(target(obs))]:
        assert not {'raw_reward', 'reward_reason', 'target_bits', 'success'} & encoded.keys()
        assert encoded['instruction'] == obs['instruction']
    after = json.loads(json.dumps(obs))
    after['elements'][0]['checked'] = True
    after['done'] = after['submitted'] = True
    row = label_transition(obs, {'kind': 'submit'}, after, .1, 'test', 0)
    assert row['success'] == 1 and row['utility'] == 1
    after['elements'][1]['checked'] = True
    assert correctness(after) == .5
    assert label_transition(obs, {'kind': 'submit'}, after, .1, 'test', 0)['success'] == 0


def test_relative_config_paths_and_resource_overlap(tmp_path, monkeypatch):
    monkeypatch.delenv('RSI_GUI_ROOT')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'run_root': 'experiment', 'resource_plan': settings.DEFAULT_RESOURCES}))
    monkeypatch.setenv('RSI_GUI_CONFIG', str(config))
    assert settings.load_settings()['run_root'] == tmp_path / 'experiment'
    for plan in [dict(settings.DEFAULT_RESOURCES, workers=[0]),
                 dict(settings.DEFAULT_RESOURCES, workers=[]),
                 dict(settings.DEFAULT_RESOURCES, rsi_policy_gpu=1),
                 dict(settings.DEFAULT_RESOURCES, workers=[1, 1])]:
        config.write_text(json.dumps({'resource_plan': plan}))
        with pytest.raises(ValueError):
            settings.load_settings()


def test_freeze_detects_code_assets_and_resource_changes(tmp_path, monkeypatch):
    from rsi_anything_gui import protocol
    monkeypatch.setattr(settings, 'ROOT', tmp_path)
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    atomic(tmp_path / 'base_manifest.json', {'unit_test_stub': {}})
    p = protocol.freeze()
    assert p['source_hashes']['jepa_anything_core/opf.py']
    assert p['source_hashes']['assets/miniwob/click-checkboxes.html']
    settings.verify_frozen()
    # A real changed task file must invalidate the frozen run.
    assets = tmp_path / 'altered_assets'
    assets.mkdir()
    (assets / 'task.html').write_text('<html>changed</html>')
    monkeypatch.setattr(settings, 'ASSETS', assets)
    with pytest.raises(RuntimeError, match='task assets differ'):
        settings.verify_frozen()


def test_frozen_configuration_rejects_different_plan(tmp_path, monkeypatch):
    from rsi_anything_gui import protocol
    monkeypatch.setattr(settings, 'ROOT', tmp_path)
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    atomic(tmp_path / 'base_manifest.json', {'unit_test_stub': {}})
    protocol.freeze()
    changed = dict(settings.SETTINGS, resource_plan=dict(settings.DEFAULT_RESOURCES, workers=[2]))
    monkeypatch.setattr(settings, 'SETTINGS', changed)
    with pytest.raises(RuntimeError, match='Frozen runtime/source changed'):
        settings.verify_frozen()


def test_policy_filter_and_terminal_replay():
    from rsi_anything_gui.policy_train import positive_rows, stratified
    def row(key, episode=1, kind='click', utility=1, success=1):
        return {'pair_hash': key, 'episode_success': episode, 'action': {'kind': kind}, 'utility': utility, 'success': success}
    rows = [row('a'), row('a'), row('b', episode=0), row('c', utility=-1), row('d', kind='submit', success=0), row('e', kind='submit')]
    positives = positive_rows(rows)
    assert [r['pair_hash'] for r in positives] == ['a', 'e']
    sample = stratified(positives, 4, random.Random(1))
    assert len(sample) == 4 and any(r['action']['kind'] == 'submit' for r in sample)
    assert all(r['pair_hash'] in ['a', 'e'] for r in sample)


def test_selected_controller_zero_uses_policy_without_forecasts():
    pytest.importorskip('playwright')
    from rsi_anything_gui.experiment import select
    proposal = {'candidate_ids': [0, 1], 'probabilities': [.6, .4],
                'actions': [{'kind': 'click', 'slot': 0}, {'kind': 'submit'}]}
    index, record = select(proposal, None, {'alpha': 0})
    assert index == 0 and record['world_used'] is False
    # An attractive submit-success logit on a click must not be used as submit value.
    forecasts = [{'submit_success_logit': 100, 'value': -.5}, {'submit_success_logit': 5, 'value': -1}]
    assert select(proposal, forecasts, {'alpha': 'pure'})[0] == 1


def test_paired_statistics_require_matching_instances(tmp_path):
    from rsi_anything_gui.statistics import paired
    a = tmp_path / 'a.jsonl'; b = tmp_path / 'b.jsonl'
    a.write_text(json.dumps({'seed': 1, 'initial_hash': 'same', 'success': 1}) + '\n')
    b.write_text(json.dumps({'seed': 1, 'initial_hash': 'different', 'success': 0}) + '\n')
    with pytest.raises(RuntimeError, match='not paired'):
        paired([a], [b])
    b.write_text(json.dumps({'seed': 2, 'initial_hash': 'same', 'success': 0}) + '\n')
    with pytest.raises(RuntimeError, match='incomplete'):
        paired([a], [b])


def test_owned_scheduler_cleanup():
    from rsi_anything_gui import orchestrate
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
    orchestrate.children['unit_owned'] = process
    try:
        orchestrate.cleanup_owned()
        assert process.poll() is not None
    finally:
        orchestrate.children.clear()
        if process.poll() is None:
            process.kill(); process.wait()


def test_scheduler_refuses_foreign_policy_port(tmp_path, monkeypatch):
    import socket
    from rsi_anything_gui import orchestrate
    monkeypatch.setattr(orchestrate, 'ROOT', tmp_path)
    atomic(tmp_path / 'protocol.json', {'source_hashes': settings.source_hashes()})
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        monkeypatch.setattr(orchestrate, 'PORT', occupied.getsockname()[1])
        with pytest.raises(RuntimeError, match='already in use'):
            orchestrate.main()
    assert not orchestrate.children
