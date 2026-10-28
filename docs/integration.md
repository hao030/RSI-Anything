# Workflow integration

RSI-Anything has one public workflow package and two task adapters.

| Layer | Location | Responsibility |
|---|---|---|
| Shared entry point | `src/rsi_anything/cli.py` | Recipes, run, resume, status, report |
| Shared runner | `src/rsi_anything/workflow/` | Stage execution, ownership, state, logs and output receipts |
| Task adapters | `src/rsi_anything/tasks/` | Stage plans and result extraction |
| Code-repair implementation | `tasks/code_repair/` | Existing scientific code, configuration and task tests |
| GUI implementation | `tasks/gui/` | Browser, models, adaptation and task tests |

## Migration

The original root code-repair source, configuration, requirements and entry scripts moved together into `tasks/code_repair/`. This preserves their relative paths and scientific implementation. The GUI directory moved from `experiments/gui/` to `tasks/gui/`; its Python namespace changed from `rsi_anything` to `rsi_anything_gui` to coexist with the shared workflow. Internal module launches, packaging and tests use the new namespace.

The shared runner invokes each task in its configured Python environment. Code repair exposes each world-update stage to the runner; GUI retains its task-native GPU scheduler for collection, adaptation and crossed evaluation. Workflow stages and task-native logs are both available in the run directory.

Original result records and upstream task assets retain their content. The earlier `integration_verification.json` records the preceding import; the reorganization checks are recorded in `workflow_verification.json`. Subsequent runtime, resume and packaging checks are recorded in `maintenance_verification.json`.

## Maintaining manifests

Regenerate the GUI release manifest after changing its files, stage the intended changes, then regenerate the repository manifest from the Git index:

```bash
python scripts/update_manifests.py --gui
git add <reviewed-paths>
python scripts/update_manifests.py --repository
git add SHA256SUMS.json
python scripts/verify_repository.py
```

Both task packages include the same five pinned JEPA core Python files. The verifier checks their equality and the presence of the browser task resources.
