"""Actual browser test independent of checkpoints and model libraries."""
from .common import ROOT, atomic, digest, correctness
from .gui_env import GUIEnv, rule_action


def smoke(episodes=4):
    if not 1 <= episodes <= 100:
        raise ValueError('Smoke episodes must be between 1 and 100')
    results = []
    env = GUIEnv()
    try:
        browser = env.browser.version
        for seed in range(190000, 190000 + episodes):
            obs = env.reset(seed)
            initial_hash = digest({k: obs[k] for k in ['instruction', 'elements']})
            steps = 0
            while not obs['done']:
                obs, _ = env.step(rule_action(obs))
                steps += 1
            success = bool(obs['submitted'] and correctness(obs) == 1.0)
            if not success:
                raise RuntimeError('Rule smoke failed on seed ' + str(seed))
            results.append({'seed': seed, 'initial_hash': initial_hash, 'steps': steps, 'success': success})
    finally:
        env.close()
    report = {'passed': True, 'browser': browser, 'episodes': results, 'model_or_training_result': False}
    atomic(ROOT / 'results/browser_smoke.json', report)
    return report
