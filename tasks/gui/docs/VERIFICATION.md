# Public source verification

| Version | Verification performed |
|---|---|
| Original experiment | Complete GPU training and experiment audit |
| Portable source package | Source, package, browser and statistical checks listed below; a new full GPU training run has not been performed |

Preparation checks:

- Original 14 experiment modules match the frozen original SHA256 records.
- 93 retained scientific/utility function bodies match after ignoring import changes; the explicit path/scheduler/protocol/auditor portability changes are listed in `source_provenance.json`.
- Retained GUI backbone method bodies match the original. JEPA core and official MiniWoB++ task assets are byte-identical copies.
- Sanitized formal endpoint rows reproduce 13 paired comparisons, their bootstrap intervals and exact paired tests, plus the original six-comparison Holm correction.
- Actual Chromium 140 browser smoke completed four excluded-probe episodes successfully.

The final local test counts, packaging and integrity results are recorded in `verification_summary.json`. Verification used a macOS CPU environment, Python 3.12.8, PyTorch 2.9.1, NumPy 2.2.4, SciPy 1.15.2 and Playwright 1.55.0.

The local test runner disabled unrelated pytest plugin auto-loading and pytest's capture plugin to work around a `readline` native-module crash. Browser contracts include actual click/replay, evaluator-input isolation, alpha-zero fallback, failed-episode filtering, injected interruption and committed-episode recovery, and probe training-data collection.

## Commands

```bash
python -m pytest
RSI_TEST_BROWSER=1 python -m pytest
python scripts/reproduce_reported_statistics.py
python scripts/verify_release.py
python -m build
```

The release manifest covers source, third-party notices, assets, tests, configs, result materials and documentation. Integrity checks scan project source/docs and verify upstream benchmark fixtures against their pinned hashes. Model locations are specified in the task configuration; checkpoints and new runtime outputs are saved in the run directories. The full experiment workflow includes model-hash verification, GPU learning preflight and a completion audit.

Project-owned code is licensed under Apache-2.0; `LICENSE`, `NOTICE` and third-party notices are included in source distributions.
