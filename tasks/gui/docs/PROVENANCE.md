# Source provenance and portability changes

The original 14 experiment modules match the frozen protocol SHA256 identities. The inherited world backbone and JEPA core came from the validated complete archive. Its archive SHA256 is `2115f9fcf0a70ab52073a9971919d8083e669a31578f3f6672e41f96ad8b7404`. Private server paths, credentials and runtime logs are excluded from this public source release.

`source_provenance.json` records original source identities, current release identities and the scope of changes. A preparation-time AST comparison checks that scientific function bodies remain unchanged except the explicit portability/audit exceptions below. JEPA core and official HTML bytes are preserved exactly.

## Changes

1. Convert flat research modules to the `rsi_anything_gui` package using relative imports.
2. Replace fixed machine paths with configuration/environment settings. Package official task assets so editable installs and wheels locate the same HTML. Use normal Playwright browser discovery.
3. Invoke child modules using the current Python interpreter. Configure physical GPU assignments; default to sequential world arms with three dedicated runtime roles.
4. Retain only the inherited world-backbone methods actually called by `GUIWorld`; unused original-domain training/scoring/checkpoint methods are omitted. Retained method bodies and parameter construction are unchanged.
5. Freeze actual package source, JEPA core, task assets, resolved runtime paths and resource settings. A changed identity requires a new run directory. Exclude download-cache bookkeeping from base input manifests, and verify required original static hashes before a public run is frozen.
6. Add exact model-revision download, environment reporting, browser smoke, package metadata, documentation, tests and release integrity checks.
7. Add cleanup of owned process groups on completion/failure/interruption.

The workflow migration updates release paths and hashes to the `rsi_anything_gui` namespace. Original experiment hashes and earlier standalone hashes remain in `source_provenance.json`; release verification checks the current file identities.

## Verification scope

Completed checks and the training scope of each version are listed in [VERIFICATION.md](VERIFICATION.md).

The reported-results files contain numeric fields from formal episode records and aggregate summaries for statistical recomputation. New experiments generate their own checkpoints, transition records and result summaries in the configured run directory.
