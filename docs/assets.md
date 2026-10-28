# Experiment assets

[Project](../README.md) · [Code Repair protocol](reproduction.en.md) · [Asset card](ASSET_CARD.md)

## Download and verify

Run from a cloned repository after `python -m pip install -e .`:

```bash
# Recorded-input evaluation: final checkpoint, evaluation data, 3B backbone
rsi assets code_repair --profile evaluate --output assets

# Full training and fresh generation: upgrade the same directory
rsi assets code_repair --profile full --output assets

# Verify every required file without downloading
rsi assets code_repair --profile full --output assets --verify
```

| Profile | Experiment archives | Base models | Use |
|---|---:|---:|---|
| `evaluate` | 64 MB | 6.2 GB | Reload the final world model and evaluate 600 recorded-input episodes |
| `full` | 300 MB total | 21.4 GB total | Rebuild data, train four stages, evaluate with fresh generation |

Allow space for extracted data and checkpoints (about 1 GB for the full profile),
download cache, the Python environment, and new runs. Models are stored directly
in `models/world/` and `models/policy/` beneath the output directory.
Archives remain in `.downloads/`; they can be removed after installation.

Archives are hosted in the [assets-v1 release](https://github.com/Gen-Verse/RSI-Anything/releases/tag/assets-v1).
The installed catalog pins each archive and every extracted file by size and SHA256.
Base models come directly from fixed Hugging Face revisions. Downloads are written
to temporary files and checked before installation. A rerun reuses complete,
verified files and archives. `evaluate` → `full` adds the missing assets.

While the repository is private, provide a repository-readable GitHub token through
`GH_TOKEN` in the process environment. For restricted Hugging Face access use
`HF_TOKEN`. Credentials are sent only to their respective API hosts.

## Offline preparation and existing assets

On a connected machine, run the installation command above and copy the entire
output directory to the target machine. Then run the same command with `--verify`.
The task guide describes the separate Python/CUDA environment.

To acquire only experiment records and checkpoints:

```bash
rsi assets code_repair --profile full --skip-models --output assets
```

Copy the exact upstream model files listed in the catalog into `assets/models/world`
and `assets/models/policy`, then rerun the normal install command. Matching files
are reused and added to the installed manifest. Use real directories when copying
models. A conflicting file produces its exact relative path; choose a clean output
directory when installing a different asset release.

The catalog is packaged at `src/rsi_anything/catalogs/code_repair.json`.
`assets/manifest.json` describes the complete installed set; it preserves full-profile
entries even when a later invocation requests only evaluation.

## GUI assets

The GUI source package includes the official task HTML and reported episode
endpoints. Its complete workflow collects browser trajectories and trains models.
Download and verify its two base models from the repository root:

```bash
.venv-gui/bin/rsi-anything-gui --config tasks/gui/configs/full.json download-models
.venv-gui/bin/rsi-anything-gui --config tasks/gui/configs/full.json verify-models
```

See the [GUI guide](../tasks/gui/README.md) for resource configuration and training.
The CPU browser smoke workflow requires no external models.

## Asset maintenance

Maintainers can produce deterministic archives from a verified native export:

```bash
python scripts/package_code_repair_assets.py \
  --source /path/to/exported-assets \
  --output /path/to/new-release \
  --provenance-prefix /path/to/original-experiment-root
```

The packager verifies original files, copies data into a separate staging tree,
normalizes location metadata, rebuilds nested checkpoint manifests, and writes
evaluation/training archives plus a catalog. The supplied prefix must cover the
original experiment root. Validate rebuilt training hashes and computational input
equality before adding a new catalog to the repository. Populate each bundle's
`url` and `api_url` from its uploaded GitHub Release asset, and retain the archive
hashes. Code, model and data terms are detailed in the [asset card](ASSET_CARD.md).
