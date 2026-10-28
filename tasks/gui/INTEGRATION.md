# GUI task in the RSI workflow

The public workflow entry point is `rsi`, installed from the repository root. The GUI runtime uses the separate `rsi_anything_gui` namespace and `rsi-anything-gui` task command. Its task implementation, assets, dependencies and tests live here.

The shared adapter lives in `../../src/rsi_anything/tasks/gui.py`. It creates the task configuration, selects the task interpreter and schedules preparation, preflight and the native adaptation/evaluation pipeline. The workflow owns `config.json`, `state.json`, `summary.json` and `logs/workflow/`; the GUI runtime owns its native protocol, models, transitions and output reports.

Recipes are in `../../configs/gui/`. Each recipe's paths resolve beside that recipe. Source and model identities are generated when the new run is frozen.

The GUI package retains its vendored JEPA core for standalone builds. Its five core Python files match the code-repair copy; repository verification checks their equality. See the [workflow design](../../docs/workflow.md) and [reorganization record](../../docs/integration.md).
