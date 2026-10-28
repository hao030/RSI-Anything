# Implementation map

| Module | Responsibility |
|---|---|
| `settings` / `cli` / `doctor` / `models` | Portable configuration, immutable identities, entry points, environment report, pinned model acquisition |
| `common` | Public state encoding, strict correctness, hashes, atomic JSON/JSONL and checkpoint identity |
| `gui_env` | Real Chromium task execution, deterministic reset, exact visible-state replay |
| `experiment` | Policy candidates, pure/hybrid selection, transactional episode commits, training-only actual browser branches |
| `world` | Original encoder and inherited backbone methods used by the GUI model |
| `gui_world` | GUI losses, OPF/standard/supervised modes, LoRA/EMA updates, critic fitting, save/load and checkpoint reuse |
| `policy_service` | Localhost policy inference, top-four terminal-inclusive candidates, adapter switching, identity-aware SQLite cache |
| `policy_train` | Deduplicated successful-episode action SFT with successful replay |
| `protocol` | Freeze cohorts, settings, source/core/assets and base manifests |
| `contracts` | Browser and actual GPU learnability preflight; excluded probe cohort |
| `worker` | World arms, held-out terminal calibration, development selection, diagnostics and confirmation |
| `rsi` | Two candidate world/policy updates, development promotion, freeze-before-test, crossed tests |
| `prepare` | Separate terminal-calibration collection |
| `orchestrate` | Owned-job scheduling, exclusive controller lease, resource checks and process-group cleanup |
| `audit` | Independent transition, action, branch, cost, dataset, checkpoint and evaluation-seal checks |
| `aggregate` | Paired uncertainty, exact tests, correction, arm/cost tables and plots |
| `smoke` | Model-free real-browser rule validation |
| `jepa_anything_core` | Unmodified vendored OPF primitives, objectives, geometry audit and baseline predictors |

## Three distinct controls

The policy ranks legal actions. Pure world control selects among the same candidates using world values and separately calibrated submission success. Hybrid control adds standardized world values to policy log probabilities with development-selected `alpha`. At `alpha=0`, hybrid control uses policy ranking directly.

## Learning data

The world encoder receives the public instruction, checkbox labels/states, candidate action and remaining budget. Evaluator reward and success form training targets after execution. Expert completions and all-legal-action branches are training supervision. Policy updates retain only positive actions from strictly successful actual episodes, with deduplication and replay. The learned state head receives current checkbox bits and the candidate click indicator; the value head receives additional public features.

## Run records

Each new run records its resolved checkpoint paths and source/input identities for auditing. Archived records retain their original identities. The included endpoint dataset supports recomputation of the reported statistics; new-run audits use the checkpoints and transition records generated in that run.
