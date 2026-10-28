# GUI experiment results

Rates below measure strict task completion. Cost columns use uncached-equivalent accounting; observed shared-cache wall time is recorded separately in the numeric summaries.

## Main workflow result

The agent reads a task instruction and visible DOM, clicks the requested checkboxes in Chromium, and submits the form. **Strict success** requires every checkbox to match the instruction and a completed submission, within 120 seconds and at most eight actions (including submission).

The same 100 RSI test instances are evaluated at three retained versions:

| Metric | Baseline | Adapted checkpoint | RSI-Anything |
|---|---:|---:|---:|
| Selected-controller success | 51/100 | 79/100 | **99/100** |
| World-based control success | 49/100 | 96/100 | **98/100** |
| Selected-controller mean actions per episode | 5.30 | 2.93 | 3.21 |
| Selected-controller repeated clicks, total | 237 | 2 | **0** |
| Selected-controller incorrect submissions, total | 9 | 21 | **1** |

**Baseline → Adapted checkpoint → RSI-Anything** follows the starting policy/world pair, the retained pair after one update, and the retained pair after two updates. Each update collects browser experience, adapts the policy from successful actions with replay, updates the world model using actual transitions and expert continuations, and selects the retained version on development instances. Evaluation uses frozen models.

**Selected controller** is the execution rule chosen on the development set; the reported runs select `alpha=0`, which follows policy ranking. **World-based control** uses world-model values to rank the same policy-proposed candidates, including submission. These are two evaluation modes for each model version.

The final selected controller completes **48 more tasks per 100** than the baseline. Mean actions include successful and failed episodes; repeated clicks and incorrect submissions are totals over the 100 episodes. The intermediate version's shorter episodes should be read alongside its 21 incorrect submissions.

## Which instances each result uses

All evaluations use the MiniWoB++ `click-checkboxes` task template. An instance is a seeded browser task, with its own instruction and checkbox configuration.

| Cohort | Instance seeds | Size | Comparison |
|---|---|---:|---|
| RSI test | 270000–270099 | 100 | Baseline, intermediate and final policy/world versions; crossed combinations |
| World test | 250000–250099 | 100 | Frozen/updated world arms with the base policy candidate pool; policy and rule baselines |
| Confirmation | 260000–260629 | 630 | The same three fixed OPF models, each before and after its world update |

The cohorts have disjoint instance seeds. The **51%** policy baseline belongs to RSI test; the **61%** policy baseline belongs to world test. All versions within a comparison use paired instances. Both RSI updates are selected and frozen before RSI test opens. The RSI workflow is one update sequence; OPF world comparisons use three fixed model training runs.

## World-model evaluation

| OPF training seed | World-test initial / 100 | World-test updated / 100 | Confirmation initial / 630 | Confirmation updated / 630 |
|---|---:|---:|---:|---:|
| 0 | 52 | 95 | 318 | 611 |
| 1 | 40 | 95 | 243 | 605 |
| 2 | 44 | 96 | 267 | 618 |
| Mean success rate | 45.33% | 95.33% | 43.81% | 97.04% |

These use pure world-based action selection. Confirmation evaluates each fixed model on the same 630 paired instances; the mean improvement is **53.23 percentage points**, with task-paired bootstrap 95% CI **[50.26, 56.30]** percentage points. Confirmation follows the prespecified world-test gain threshold in the [protocol](PROTOCOL.md).

## Reading crossed evaluations and metrics

For each RSI update, four combinations are evaluated: old policy/old world, old policy/new world, new policy/old world, and new policy/new world. In file identifiers such as `rsi_g1_old_new_pure`, the first position denotes the **policy** and the second the **world**. `g1` and `g2` identify the two successive updates; “old” means the retained version immediately before that update. `selected` and `pure` identify the two action-selection modes defined above.

- **Success:** the fraction of complete episodes satisfying the strict task criterion.
- **Mean steps:** clicks plus submission, averaged over all evaluated episodes.
- **Repeated clicks:** clicks beyond the first click on each checkbox within an episode, summed across episodes.
- **Wrong submissions:** episodes that submitted but failed the strict success criterion, summed across episodes.
- **Equivalent seconds/episode:** policy time charged at its uncached equivalent plus measured world/browser time, divided by the episode count. Observed wall time is retained separately.

<details>
<summary>All 42 formal 100-instance endpoints and exact arm identifiers</summary>


| Arm | n | Success | Mean steps | Repeated clicks | Wrong submissions | Equivalent seconds/episode |
|---|---:|---:|---:|---:|---:|---:|
| `opf_seed0_frozen_pure` | 100 | 52% | 5.22 | 260 | 3 | 1.134 |
| `opf_seed0_frozen_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.179 |
| `opf_seed0_updated_pure` | 100 | 95% | 3.57 | 63 | 0 | 0.771 |
| `opf_seed0_updated_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.180 |
| `opf_seed1_frozen_pure` | 100 | 40% | 6.01 | 315 | 2 | 1.269 |
| `opf_seed1_frozen_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.179 |
| `opf_seed1_updated_pure` | 100 | 95% | 3.47 | 49 | 0 | 0.740 |
| `opf_seed1_updated_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.180 |
| `opf_seed2_frozen_pure` | 100 | 44% | 5.53 | 279 | 2 | 1.165 |
| `opf_seed2_frozen_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.180 |
| `opf_seed2_updated_pure` | 100 | 96% | 3.08 | 20 | 0 | 0.642 |
| `opf_seed2_updated_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.180 |
| `ce_seed0_frozen_pure` | 100 | 41% | 5.25 | 241 | 13 | 1.129 |
| `ce_seed0_frozen_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.179 |
| `ce_seed0_updated_pure` | 100 | 95% | 3.04 | 19 | 0 | 0.650 |
| `ce_seed0_updated_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.179 |
| `standard_seed0_frozen_pure` | 100 | 6% | 7.75 | 471 | 0 | 1.574 |
| `standard_seed0_frozen_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.179 |
| `standard_seed0_updated_pure` | 100 | 98% | 3.36 | 43 | 1 | 0.685 |
| `standard_seed0_updated_selected` | 100 | 61% | 4.82 | 227 | 7 | 0.179 |
| `baseline_policy` | 100 | 61% | 4.82 | 227 | 7 | 0.181 |
| `baseline_rule` | 100 | 100% | 2.87 | 0 | 0 | 0.010 |
| `rsi_g1_old_old_pure` | 100 | 49% | 5.60 | 258 | 1 | 1.314 |
| `rsi_g1_old_old_selected` | 100 | 51% | 5.30 | 237 | 9 | 0.337 |
| `rsi_g1_old_new_pure` | 100 | 95% | 3.88 | 59 | 1 | 0.901 |
| `rsi_g1_old_new_selected` | 100 | 51% | 5.30 | 237 | 9 | 0.336 |
| `rsi_g1_new_old_pure` | 100 | 50% | 5.59 | 255 | 0 | 1.201 |
| `rsi_g1_new_old_selected` | 100 | 79% | 2.93 | 2 | 21 | 0.128 |
| `rsi_g1_new_new_pure` | 100 | 96% | 3.83 | 53 | 0 | 0.818 |
| `rsi_g1_new_new_selected` | 100 | 79% | 2.93 | 2 | 21 | 0.127 |
| `rsi_g1_old_policy` | 100 | 51% | 5.30 | 237 | 9 | 0.336 |
| `rsi_g1_new_policy` | 100 | 79% | 2.93 | 2 | 21 | 0.127 |
| `rsi_g2_old_old_pure` | 100 | 96% | 3.83 | 53 | 0 | 0.815 |
| `rsi_g2_old_old_selected` | 100 | 79% | 2.93 | 2 | 21 | 0.127 |
| `rsi_g2_old_new_pure` | 100 | 98% | 3.76 | 48 | 2 | 0.808 |
| `rsi_g2_old_new_selected` | 100 | 79% | 2.93 | 2 | 21 | 0.127 |
| `rsi_g2_new_old_pure` | 100 | 98% | 3.82 | 50 | 0 | 0.806 |
| `rsi_g2_new_old_selected` | 100 | 99% | 3.21 | 0 | 1 | 0.140 |
| `rsi_g2_new_new_pure` | 100 | 98% | 3.78 | 48 | 2 | 0.804 |
| `rsi_g2_new_new_selected` | 100 | 99% | 3.21 | 0 | 1 | 0.139 |
| `rsi_g2_old_policy` | 100 | 79% | 2.93 | 2 | 21 | 0.127 |
| `rsi_g2_new_policy` | 100 | 99% | 3.21 | 0 | 1 | 0.140 |

The world-test and RSI-test endpoints use different 100-instance cohorts. World arms use the base policy candidate pool; RSI crosses hold either the old/new world or old/new policy fixed. `selected` means development-selected hybrid/fallback; all original selected alphas were zero. `pure` selects by world values among the same policy-proposed candidates.

</details>

## Paired comparisons

| Comparison | Gain (percentage points) | Task-paired bootstrap 95% CI |
|---|---:|---:|
| Pure OPF update, 100 instances | 50.00 | [42.67, 57.33] |
| Selected world update, 100 instances | 0.00 | [0.00, 0.00] |
| Pure OPF confirmation, 630 instances | 53.23 | [50.26, 56.30] |
| RSI proposal 1 policy-only | 28.00 | [18.00, 38.00] |
| RSI proposal 2 policy-only | 20.00 | [13.00, 28.00] |

Paired bootstrap resamples task instances across the three fixed OPF models. Six prespecified secondary comparisons receive Holm adjustment; supplemental RSI contrasts are descriptive.

## Confirmation endpoint counts

| Fixed OPF model | Initial successes / 630 | Updated successes / 630 |
|---|---:|---:|
| Seed 0 | 318 | 611 |
| Seed 1 | 243 | 605 |
| Seed 2 | 267 | 618 |

## Evaluation setup

- Task: MiniWoB++ `click-checkboxes`, with visible-DOM observations, a 120-second timeout and an eight-action budget.
- Training: expert demonstrations, actual browser branches and expert continuations. World updates train the online LoRA encoder, EMA target where applicable, predictive factors and learned heads.
- RSI evaluation: one policy training seed and 100 paired test instances; the final retained policy achieves 99% task completion.
- OPF confirmation: three fixed world models, each evaluated before and after updating on 630 paired instances.

## Recompute statistics

The endpoint dataset contains 7,980 rollout records across 48 files, with repeated evaluations on paired cohorts: 2,000 world-arm tests, 2,000 RSI crossed/policy tests, 200 baseline tests and 3,780 confirmation rollouts.

Run `python scripts/reproduce_reported_statistics.py` to recompute 13 paired comparisons, exact paired tests and the six-comparison Holm correction from these records.
