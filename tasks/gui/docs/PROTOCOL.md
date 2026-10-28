# Frozen scientific protocol

The complete original scientific settings are in `original_scientific_protocol.json`. The public `freeze` command retains those settings and emits a new public-run source/input/resource identity.

| Cohort | Instance seeds | Size |
|---|---|---:|
| Excluded preflight probe | 190000–190003 | 4 |
| Initial training | 200000–200099 | 100 |
| World update acting | 210000–210099 | 100 |
| Terminal calibration | 220000–220049 | 50 |
| Development and selection | 230000–230049 | 50 |
| World diagnostics | 240000–240049 | 50 |
| World test | 250000–250099 | 100 |
| Conditional OPF confirmation | 260000–260629 | 630 |
| RSI test | 270000–270099 | 100 |
| RSI update proposal 1 acting | 280000–280099 | 100 |
| RSI update proposal 2 acting | 290000–290099 | 100 |

All cohorts use the `click-checkboxes` task template with disjoint instance seeds. Both RSI proposals are trained/selected/frozen before the RSI test is opened.

## Model and controller settings

- Five world arms: OPF seeds 0/1/2, supervised control seed 0, standard JEPA seed 0. Initial and update worlds use three joint epochs followed by 100 completion-critic epochs, batch size 128, head learning rate 0.001.
- Encoder: Qwen2.5-Coder-3B-Instruct; compact public text, last effective token pooling, limit 1024 tokens. LoRA rank 16, alpha 32, dropout 0, q/k/v/o projection targets. EMA decay 0.996.
- OPF: latent width 256, four groups of width 64, learned soft-Gram analysis projectors; general synthesis in the unchanged vendored core.
- Optimizer groups: encoder LoRA 0.000005; representation 0.0001; OPF 0.00001; readouts 0.001; learned per-slot dynamics 0.01. Weight decay 0.01.
- Training targets use actual next public state, actual expert-continuation discounted outcomes (discount 0.9), immediate progress and strict terminal success. Expert assistance is confined to training.
- Terminal calibration: temperatures 0.5/1/2/4 and biases −2/−1/0/1/2, selected by Brier score on its separate cohort.
- Hybrid candidates: alpha 0.25 or 1.0. Enable if development success gain is at least 0.04 and uncached-equivalent cost is at most 4× policy baseline; otherwise fallback to alpha 0.
- Promote if development success gain is at least 0.04 and equivalent cost is at most 1.2× incumbent. Rejected candidates still receive crossed evaluation and are reported.
- Conditional confirmation runs when mean pilot pure-OPF gain is at least 0.05; all three fixed OPF models then compare initial/updated worlds on 630 instances.
- Two RSI proposals, one training seed. Policy SFT uses strictly successful actual episodes, positive action deduplication, up to 256 new plus equal replay, one epoch, batch size 4, AdamW learning rate 0.00001. If no positives exist, record no update.

The full experiment uses the cohort sizes and thresholds above. Configure runtime paths and GPU assignments in the task configuration. For an exploratory study with different settings, record the revised protocol in a new run directory. Use the browser smoke recipe for a short environment check.
