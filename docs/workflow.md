# RSI workflow

A workflow recipe selects a task, runtime, output directory and task options. The shared runner executes the recipe's stages and records their results. Task adapters translate the recipe into concrete commands.

## Commands

```bash
rsi tasks
rsi plan --config configs/gui/default.json
rsi run --config configs/gui/default.json
rsi run --config configs/gui/default.json --resume
rsi status --run runs/gui
rsi report --run runs/gui
```

`plan` prints commands and resolved paths without creating a run or starting workers. `run` executes stages in order. `status` reads the state, and `report` combines it with actual task result files. Evaluation-only code repair uses `configs/code_repair/evaluate.json`.

## Recipes

| Field | Meaning |
|---|---|
| `task` | `code_repair` or `gui` |
| `mode` | Code repair: `full`, `evaluate`, `verify`; GUI: `full`, `smoke` |
| `run_root` | Directory for this run |
| `python` | Worker interpreter path, executable name, or `current` |
| `options` | Task-specific options |
| `repository` | Optional path to the source checkout |

Relative paths resolve beside the recipe. Virtual environment executable paths retain their symlinks so the selected environment remains active. Install the common workflow from the source checkout with `pip install -e .`.

Code-repair options: `assets`, `train_gpu`, `gpus`, `inputs`, `judge_workers`, and an optional `checkpoint` for evaluation. Full mode uses its newly trained final checkpoint. GUI options: `world_model`, `policy_model`, `assets`, `policy_port`, `resource_plan`, `download_models` and smoke `episodes`.

## Stage plans

**Code repair, full:** verify assets → update initial world → update semantic world → update expanded world → update incumbent world → evaluate. Each update receives the preceding stage's checkpoint. Training consumes the recorded training transitions. Data reconstruction is available through the native `prepare` command.

**GUI, full:** acquire models → check environment → freeze protocol → browser preflight → world preflight → adapt and evaluate. The final stage runs the task-native collection, baseline/world comparisons, two policy/world updates, conditional confirmation, audit and aggregation. Set `download_models` to false to use prepared models.

**GUI, smoke:** execute real browser episodes with the rule controller and collect the task report.

## Task adapter contract

Each task implements two functions:

- `plan(config) -> Plan`: stages plus rendered task configuration files.
- `results(run_root) -> dict`: metrics and artifact paths from actual task outputs.

Each `Stage` names its command, working directory, environment, expected output files or checkpoint directories and any outputs to archive before a retry. The common runner handles execution, exclusive ownership, process cleanup, progress and persistence. Scientific training and action-selection operations remain in the task implementations.

## Run records

```text
runs/<name>/
├── config.json             # Resolved workflow recipe
├── state.json              # Stage status, attempts and output hashes
├── summary.json            # Task metrics and workflow status
├── logs/workflow/          # One log per stage attempt
├── attempts/               # Retained incomplete code-repair stage outputs
└── ...                     # Task-native checkpoints and evaluation records
```

The code-repair workflow stores checkpoints under `checkpoints/world/<stage>/` and evaluation under `evaluations/code_repair/`. GUI retains `data/`, `checkpoints/`, `results/` and `outputs/` and receives a generated `task-config.json`.

Resume requires the same recipe and source identity. Completed stages are skipped after checking their recorded output hashes, including all files within declared checkpoint directories. Source identity includes the browser HTML, scripts, styles and media. An incomplete code-repair output directory is moved into `attempts/` before retry; GUI restarts its native stage and uses native checkpoint/episode reuse. Changed scientific inputs use a new run root. A filesystem lock prevents two runners from owning the same run.

## Task environments

The runner uses only the Python standard library. Tasks run in their own Python interpreters. GUI workers use `rsi_anything_gui`; the workflow uses `rsi_anything`. Code repair retains its internal imports under its task directory. Both tasks retain their pinned dependencies and JEPA core.
