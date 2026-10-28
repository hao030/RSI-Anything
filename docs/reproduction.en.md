# Code-repair reproduction

[Project](../README.md) · [简体中文](reproduction.md)

Run the commands below from the repository root. This guide covers checkpoint evaluation, training-data reconstruction and the four-stage training chain.

The evaluation uses 200 CodeContests problems with seeds 0, 1 and 2, giving 600 problem–seed episodes. Each run writes its measured success rate and per-seed results to `summary.json`.

## Environment and assets

Use Linux and Python 3.10. Dependency versions are pinned in `tasks/code_repair/requirements.txt`; the recorded environment is in `tasks/code_repair/config/environment.json`. The program executor requires `runuser`, `unshare` and the `nobody` account for isolated execution and stdout-based judging.

```bash
python3.10 -m venv .venv-code-repair
source .venv-code-repair/bin/activate
python -m pip install -r tasks/code_repair/requirements.txt
python -m pip install -e .

# Final checkpoint, evaluation inputs and 3B world-model backbone
rsi assets code_repair --profile evaluate --output assets

# Training records, reference checkpoints and both base models
rsi assets code_repair --profile full --output assets
rsi assets code_repair --profile full --output assets --verify
```

Both profiles use the same directory; upgrading from `evaluate` to `full` reuses verified files. The evaluation archive is about 64 MB and the training supplement is about 236 MB. Base models occupy about 6.2 GB for evaluation or 21.4 GB for the full profile. Allow additional space for extracted files, caches and new runs. See the [asset guide](assets.md) for downloads and offline preparation, and the [asset card](ASSET_CARD.md) for provenance and terms.

`assets/manifest.json` records installed file sizes, SHA256 hashes and sources. Training provenance uses portable identifiers; the configuration pins hashes for both original and released training records.

To export an existing experiment directory:

```bash
python tasks/code_repair/export_assets.py --source-root /path/to/experiment-source --output /path/to/portable-assets --mode copy
python tasks/code_repair/reproduce.py verify-assets --assets /path/to/portable-assets
```

Use `--mode symlink` to reuse local storage, or `--mode copy` when transferring assets to another machine.

## Evaluate 600 episodes with recorded inputs

```bash
python tasks/code_repair/reproduce.py evaluate \
  --assets assets \
  --gpus 0,1,2,3,4,5,6,7 \
  --inputs recorded \
  --output runs/evaluation
```

The command reloads the final JEPA world model, scores four revisions and the current program at each step, and selects the next program. After all 600 decisions have been completed and sealed, it runs hidden tests and summarizes the results.

`recorded` fixes the initial programs, first-round candidates, previously sampled later candidates and public execution feedback. Execution durations are also model inputs and are preserved. World-model scores and decisions are recomputed in the new run. A state missing from the recorded inputs stops evaluation with an error identifying the missing state.

| Output | Contents |
| :--- | :--- |
| `protocol.json` | Model identity, input mode, method configuration and provenance hashes |
| `public/*.jsonl` | Recomputed scores and decisions |
| `public_seal.json` | Seal for all 600 decisions |
| `results.jsonl` | Hidden-test judging results |
| `summary.json` | Measured successes, per-seed results and reference-decision agreement |
| `complete.json` | Summary checksums |

`successes` and `per_seed_successes` record measured judging outcomes. `reference_decisions_matched` records agreement with the reference decisions.

## Reconstruct training data

```bash
python tasks/code_repair/reproduce.py prepare --assets assets --output runs/prepared
```

This reconstructs the three update datasets from recorded TRAIN execution transitions, compares their SHA256 hashes with the pinned training files, and rebuilds complete candidate-group batches. Public-test and full-suite execution records feed their respective readout heads.

| Stage | Training records | Configuration |
| :--- | ---: | :--- |
| Initial JEPA | 32,920 | Seed 0; native OPF; 1 epoch |
| Execution-quality adaptation | 7,872 | Seed 1; readout heads for 1 epoch, then joint training for 1 epoch; distillation weight 0.25 |
| Expanded training | 41,088 | Seed 1; joint training for 1 epoch; distillation and group-ranking weights 1 |
| Incumbent-paired update | 22,192 | Seed 0; 1 epoch; pairwise weight 4; full-suite anchoring weight 0; half learning rate |

Training starts from recorded execution transitions. Asset hashes fix the original contexts, labels and durations.

## Run the full training chain

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/code_repair/reproduce.py train \
  --assets assets --stage all --output runs/training
```

The four stages run sequentially: initial JEPA, execution-quality adaptation, expanded training and the incumbent-paired update. Each stage inherits the checkpoint newly trained by the previous stage. Development-selected settings are stored in `tasks/code_repair/config/experiment.json`.

To train one stage from its verified reference parent:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/code_repair/reproduce.py train \
  --assets assets --stage incumbent --output runs/incumbent
```

Evaluate the newly trained checkpoint:

```bash
python tasks/code_repair/reproduce.py evaluate \
  --assets assets --checkpoint runs/incumbent/joint_checkpoint \
  --inputs fresh --gpus 0,1,2,3,4,5,6,7 --output runs/retrained-evaluation
```

`fresh` preserves the initial programs and first-round candidates, generates subsequent candidates again, and reruns public tests. New trajectories, execution records and measured results are saved in the specified output directory.

## Implementation

`tasks/code_repair/src/world.py` and `tasks/code_repair/vendor/jepa-anything-core/` implement native JEPA encoding, latent prediction, OPF and EMA. `semantic_world.py` supplies public/full-suite readout heads and group ranking; `losses.py` implements preferences relative to the incumbent program. `grouping.py` and `incumbent.py` preserve complete candidate groups and batches containing four distinct tasks.

Selection uses `P(full-suite pass) + 0.25 × predicted pass fraction`. A candidate replaces the incumbent only when its score is strictly greater; ties retain the incumbent. Each round has four revisions and five world-model predictions, with at most three repair rounds. Evaluation performs no optimizer updates.

## Verification

```bash
python -m unittest discover -s tasks/code_repair/tests -v
python tasks/code_repair/verify_result.py --assets assets --output runs/evaluation
CUDA_VISIBLE_DEVICES=0 python tasks/code_repair/reproduce.py train \
  --assets assets --stage incumbent --max-steps 3 \
  --no-development --output runs/training-smoke
```

Short training checks exercise backpropagation, checkpoint saving and reloading. Packaging-time validation is recorded in `tasks/code_repair/config/validation.json`:

| Check | Recorded validation |
| :--- | :--- |
| Evaluation | 600 model decisions and hidden-test episodes; decisions matched the reference |
| Training data | All three reconstructed update datasets matched the original hashes |
| Training chain | Three steps in each of four stages; zero checkpoint reload error |
| Full-epoch retraining | Not run during source packaging |
| Reference-checkpoint geometry | Strict orthogonality and factor-independence checks did not pass |

Full training saves geometry diagnostics for each stage.

## Source provenance

Core-function provenance and hashes are in `tasks/code_repair/config/source_provenance.json`. The JEPA-Anything core revision is `c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`; its upstream license is retained in `tasks/code_repair/vendor/LICENSE`.

## Shared workflow

The root `rsi` command runs four-stage training and evaluation using `configs/code_repair/default.json`, or evaluates an existing checkpoint using `configs/code_repair/evaluate.json`. Asset paths default to the repository-root `assets/` directory.

```bash
rsi plan --config configs/code_repair/default.json
rsi run --config configs/code_repair/default.json
rsi report --run runs/code-repair
```

The workflow passes newly trained checkpoints between stages and sends the final checkpoint to evaluation. Task-native commands above also support individual training, reconstruction and evaluation stages. See [workflow design](workflow.md) for stage interfaces.
