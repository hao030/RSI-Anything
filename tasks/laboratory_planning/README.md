# Laboratory Planning

**Turn experimental objectives into executable laboratory plans with RSI-Anything.**

[简体中文](README.zh-CN.md) · [RSI-Anything](../../README.md)

<p>
  <a href="https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning">
    <img src="https://img.shields.io/badge/Hugging%20Face-RSI--Anything-FFB000?style=plastic&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face · RSI-Anything" height="28">
  </a>
</p>

LabHorizon Level 2 asks an agent to plan a sequence of laboratory operations from an experimental context, a goal, available materials, constraints and an action library. A plan specifies which operations to perform, their parameters and how intermediate products pass between steps.

JEPA-Anything guides the choice of laboratory plans. A planning agent proposes four candidates, and the world model forecasts their quality, improvement, risk and uncertainty to select the next plan.

## RSI workflow

RSI improves laboratory planning through repeated updates to a **planning policy and a JEPA world model**. Each round starts from the retained pair and uses task feedback to improve candidate generation and selection.

### Learning rounds

Starting with a trained initial pair, each round follows five steps:

1. **Generate candidates.** The policy reads the task and proposes one greedy plan (`a1`) and three sampled plans (`a2`–`a4`). The greedy plan is the reference for predicted improvement.
2. **Predict and select.** Check plan structure and rank legal candidates by predicted quality and improvement, with penalties for risk and uncertainty. Select the highest-scoring legal plan.
3. **Collect candidate feedback.** On learning tasks, score all four candidates with the task evaluator and record action-order, parameter-accuracy and intermediate-product-dependency feedback together with the final planning score.
4. **Update both models.** For each task, use the highest-scoring legal candidate to construct policy training examples and continue training from the retained policy. Refit the world model on accumulated task–candidate–evaluation records to form a new pair for validation.
5. **Validate and retain.** Evaluate the new pair together on validation tasks, comparing planning quality and structural validity with the retained pair. A pair that passes acceptance becomes the starting point for the next round; otherwise the current pair is retained.

**Recursion comes from carrying the accepted pair and accumulated experience into the next learning round.** The retained policy generates new candidates, and their evaluated outcomes on learning tasks extend the training data. World-model selections are used in joint validation, and the accepted pair becomes the starting point for the next improvement step.

### Freeze and evaluate

**The validation set and the F200 test set are separate splits with no overlapping tasks.** Learning tasks provide model-update examples, and validation tasks determine which policy–world-model pair is retained at each round. After learning and acceptance, freeze the retained pair and evaluate it on **LabHorizon F200**. F200 is reserved for evaluation and result reporting after the models are frozen.

The published commands cover the **frozen inference stage**:

- `infer.py world` generates four plans with a local planning adapter and selects one with the matching JEPA world checkpoint.
- `infer.py select` applies the same world-model selection to four supplied candidate plans.

Outputs contain the selected plan and its forecasts. The `forecast.feedback` fields are predictions of evaluation outcomes; the learning rounds use feedback from the task evaluator.

## Setup

Use Python 3.10 or newer. Run commands from the repository root:

```bash
python -m pip install -r tasks/laboratory_planning/requirements-world.txt huggingface-hub
export MODEL_DIR="$PWD/models/laboratory_planning"
```

| Inference component | Backbone or architecture | Adapter or checkpoint |
| :--- | :--- | :--- |
| JEPA candidate generation | [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | Local P4 adapter for seed 1401, 1402 or 1403 |
| JEPA world model | JEPA-Anything | [Hugging Face checkpoints](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning/checkpoints) |

Download the JEPA world checkpoints:

```bash
hf download Gen-Verse/RSI-Anything --repo-type dataset \
  --include "laboratory_planning/checkpoints/*" \
  --local-dir "$MODEL_DIR/rsi-anything"
```

## Quick start: select a plan

Provide JSONL records containing `task` and four `candidates`, ordered `a1`, `a2`, `a3`, `a4`. Each candidate has a `candidate_id` and `completion`; `a1` is the reference plan. See the [candidate example](examples/candidates.jsonl).

```bash
python tasks/laboratory_planning/infer.py select \
  --world-dir "$MODEL_DIR/rsi-anything/laboratory_planning/checkpoints/seed1401" \
  --input tasks/laboratory_planning/examples/candidates.jsonl \
  --output outputs/selected_plans.jsonl
```

Candidate selection runs on CPU. Outputs include the selected candidate, final `completion`, all candidates and a `forecast` with predicted quality, improvement, risk, uncertainty and validity. Add `--include-factors` to return the JEPA latent coordinates.

## Task inputs

Use JSONL with one task per line:

| Field | Content |
| :--- | :--- |
| `id` | Task identifier |
| `context` | Experimental background |
| `goal` | Requested experimental outcome |
| `constraints` | List of constraints |
| `available_inputs` | JSON-encoded string describing named materials |
| `action_pool` | Python definitions of the available operations |

The [task example](examples/tasks.jsonl) illustrates the input format. Plans use Python calls to the supplied operations, with keyword arguments and named intermediate products.

## Generate candidates with a local planning adapter

Generate four candidates with a local P4 planning adapter and select a plan with its matching JEPA checkpoint:

```bash
python -m pip install -r tasks/laboratory_planning/requirements.txt
python tasks/laboratory_planning/infer.py world \
  --base-model Qwen/Qwen3.5-4B-Base \
  --adapter "$MODEL_DIR/seed1401/P4" \
  --world-dir "$MODEL_DIR/rsi-anything/laboratory_planning/checkpoints/seed1401" \
  --seed 1401 \
  --input tasks/laboratory_planning/examples/tasks.jsonl \
  --output outputs/jepa_plans.jsonl \
  --device cuda:0
```

The policy produces one greedy candidate and three sampled candidates. JEPA selection combines predicted quality and improvement with risk, uncertainty and structural validity. The planning policy and world model remain frozen during inference.

Use the corresponding adapter, world checkpoint and seed for 1402 or 1403.

## Code

| File | Purpose |
| :--- | :--- |
| [infer.py](infer.py) | Inference command line |
| [generator.py](src/lab_inference/generator.py) | Candidate plan generation |
| [world.py](src/lab_inference/world.py) | JEPA forecasts and candidate selection |
| [features.py](src/lab_inference/features.py) | Task text and plan-structure features |
| [postprocess.py](src/lab_inference/postprocess.py) | Candidate plan normalization |

JEPA-Anything components retain [Apache-2.0](licenses/JEPA-Anything.LICENSE); the LabHorizon plan parser retains [MIT](licenses/LabHorizon-parser.LICENSE).
