# Suzuki Reaction Optimization

Select the next Suzuki–Miyaura reaction condition using a calibrated ReactionT5–JEPA world-model prior and measured yield feedback.

[简体中文](README.zh-CN.md) · [RSI-Anything](../../README.md)

<p>
  <a href="https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/reaction_optimization">
    <img src="https://img.shields.io/badge/Hugging%20Face-RSI--Anything-FFB000?style=plastic&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face · RSI-Anything" height="28">
  </a>
</p>

## Results

The frozen Suzuki adapter was evaluated by sequentially replaying measured yields from a flow LC–MS screen. Each run starts with eight measured conditions and has a total budget of 40 distinct observations, including the initial eight.

| Metric | Result |
| :--- | ---: |
| Candidate conditions | 321 |
| Runs, seeds 4001–4100 | 100 |
| Mean best observed yield at observation 10 | **96.15%** |
| Mean final best observed yield at observation 40 | **96.15%** |
| Runs finding the candidate-pool optimum by observation 10 | **100 / 100** |
| Mean first observation finding the optimum | **9.94** |

96.15% is an observed reaction yield. The evaluation concerns one fixed reaction context and its screened condition pool. The machine-readable [evaluation record](results/evaluation.json) includes the protocol, selected calibration and artifact identities; [release validation](results/validation.json) records checks of the packaged inference code.

## Setup and quick start

Use Python 3.10 or newer. Run these commands from the repository root:

```bash
python -m pip install -r tasks/reaction_optimization/requirements.txt
python tasks/reaction_optimization/infer.py \
  --observations tasks/reaction_optimization/examples/observations.json \
  --output runs/suzuki/next_condition.json
```

The included example has **synthetic yields** to demonstrate the input format. Replace them with your measurements. Omit `--output` to print the recommendation as JSON. Output files are created without overwriting existing files.

The released prediction table runs on CPU. It contains the world model's frozen, adapted predictions for this pool, so selection can run directly after installing NumPy and SciPy.

## Inputs and feedback

Supply a JSON list of observed conditions:

```json
[
  {"candidate_index": 0, "yield": 42.0},
  {"candidate_index": 1, "yield": 55.0}
]
```

| Field | Meaning |
| :--- | :--- |
| `candidate_index` | Integer in 0–320, identifying a row of the released candidate catalog. |
| `yield` | Measured yield as a percentage in 0–100, such as `55.0`. |

Start with a nonempty list. The reported protocol uses eight initial observations. Candidate IDs must be unique. After measuring the recommended condition, append its `candidate_index` and measured yield to the list and call inference again. Feedback changes the best observed yield used by expected improvement and removes measured conditions from selection. Predictions stay frozen during this loop.

The [candidate catalog and predictions](assets/suzuki_prior.json) cover electrophile **1c, 6-OTf-Q**, nucleophile **2b, Boronic Ester** and catalyst **Pd(OAc)2**, with 321 ligand/base/solvent combinations. Each record includes the complete condition names. `original_table_index` provides the row mapping into the 5,030-row filtered screening table; use `candidate_index` when submitting observations.

## Outputs

| Field | Meaning |
| :--- | :--- |
| `candidate_index` | Next unobserved condition to measure. |
| `original_table_index` | Corresponding filtered screening-table row. |
| `conditions` | Electrophile, nucleophile, catalyst, ligand, base and solvent names. |
| `predicted_yield_percent` | Frozen adapted mean yield. |
| `calibrated_std_percent_points` | Calibrated predictive standard deviation, in yield percentage points. |
| `expected_improvement` | Acquisition score in yield percentage points. |
| `observations_used` | Number of measurements supplied. |
| `best_observed_yield_percent` | Highest yield in the supplied observations. |
| `model_id` | Released Suzuki prediction-table identity. |

The [example recommendation](examples/recommendation.json) is generated from the synthetic input above. Selection returns one condition at a time, with equal scores resolved by the smaller candidate ID. Inference stops with a clear error when all candidates have been measured.

## Python API

From the repository root:

```python
import json
import sys

sys.path.insert(0, "tasks/reaction_optimization/src")
from suzuki_inference import load_prior, recommend

prior = load_prior()
with open("tasks/reaction_optimization/examples/observations.json") as source:
    observations = json.load(source)
next_condition = recommend(observations, prior)
print(next_condition["conditions"])
```

Reuse `prior` for successive recommendations. `load_prior(path)` and `--prior path` accept a prediction table using the same schema.

## Verification and files

```bash
python tasks/reaction_optimization/scripts/verify_assets.py
python -m unittest discover -s tasks/reaction_optimization/tests -v
```

| File | Purpose |
| :--- | :--- |
| [infer.py](infer.py) | Portable command-line entry point. |
| [prior.py](src/suzuki_inference/prior.py) | Prediction-table and condition-catalog validation. |
| [selector.py](src/suzuki_inference/selector.py) | Expected improvement and sequential recommendation. |
| [cli.py](src/suzuki_inference/cli.py) | Observation input and JSON output. |
| [suzuki_prior.json](assets/suzuki_prior.json) | Frozen calibrated predictions and candidate conditions. |
| [manifest.json](assets/manifest.json) | Prediction asset checksum and source identity. |
| [evaluation.json](results/evaluation.json) | Current model's measured results and evaluation protocol. |

Project-owned code follows the repository's [Apache-2.0 license](../../LICENSE). ReactionT5 and JEPA component notices are retained in [licenses](licenses/).
