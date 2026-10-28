# Benchmark settings and results

[Project](../README.md) · [简体中文](benchmarks.zh-CN.md)

RSI-Anything connects candidate generation, predicted outcomes and interaction feedback across five domains. This page describes the paper's code and GUI benchmark comparisons.

## Code repair

The agent starts from a program and revises it using test feedback. ReasonFlux-Coder-7B proposes four repair candidates per round, for at most three rounds. The world model scores candidate execution outcomes to guide selection. CodeContests uses 200 problems and three seeds, giving 600 problem–seed evaluations; final success requires passing all hidden tests.

### APPS Introductory and LiveCodeBench

These experiments test the repair workflow on two additional problem collections. For each problem and seed, methods share the same initial program and ReasonFlux-Coder-7B policy. Public-test feedback guides revisions; repair stops when all public tests pass. Final success requires passing the complete official test suite.

| Setting | APPS Introductory | LiveCodeBench v6 |
| :--- | :--- | :--- |
| Problems | 1,000 | 1,055 |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Problem–seed evaluations | 3,000 | 3,165 |
| Candidates per round | 4 | 4 |
| Maximum repair rounds | 3 | 3 |
| RSI-Anything full-test success | **46.37%** | **33.05%** |

The reported metric is the arithmetic mean of full-test success rates across the three seeds. RSI-Anything exceeds random selection and the capacity-matched static verifier under the shared coder and candidate budget. Figure 2, panels **e–f**, presents these comparisons; panels **a–d** show the workflow, CodeContests outcomes, prediction quality and feedback use.

[View Figure 2](media/figure2-code-repair.pdf) · [CodeContests reproduction](reproduction.en.md)

## GUI interaction

The agent observes an interface, proposes actions, uses predicted outcomes to select an action, and receives task feedback from the environment. Improvements retained from interaction guide subsequent decisions.

### MiniWoB++

The browser workflow uses `click-checkboxes`: select the checkboxes named in the instruction and submit the form. Evaluation uses the same 100 test instances across three retained controller versions, with a limit of 120 seconds and eight actions per episode.

| Metric | Initial | Intermediate | Final |
| :--- | ---: | ---: | ---: |
| Task success | 51% | 79% | **99%** |
| Mean actions per episode | 5.30 | 2.93 | **3.21** |
| Episodes with incorrect submissions | 9 | 21 | **1** |

The final controller reduces mean actions by **39.4%** relative to the initial controller; repeated clicks decrease from **237 to 0**. The intermediate controller takes fewer actions but has more incorrect submissions, so action counts are interpreted alongside task success and submission errors.

[GUI workflow](../tasks/gui/README.md) · [Evaluation cohorts and complete statistics](../tasks/gui/docs/RESULTS.md)

### OSWorld

OSWorld evaluates GUI agents on desktop tasks. The paper compares the initial agent, RLAnything and RSI-Anything using task success. Predictions guide action selection, and environment feedback supports improvements carried into subsequent interactions.

| Method | Success-rate gain relative to the initial agent |
| :--- | ---: |
| Initial agent | Reference |
| RLAnything | +12.3% |
| RSI-Anything | **+18.4%** |

RSI-Anything also improves success by **5.5% relative to RLAnything**. Relative gain is `100 × (method success / reference success − 1)%`; the table uses the initial agent as the common reference, while the 5.5% comparison uses RLAnything as its reference.

## Implementation guides

| Workflow | Entry point |
| :--- | :--- |
| Code repair: CodeContests training and evaluation | [Environment, assets and reproduction](reproduction.en.md) |
| GUI: MiniWoB++ browser training and evaluation | [Task guide](../tasks/gui/README.md) |

The [project overview](../README.md#domains) also covers laboratory planning, reaction optimization and immune science, with their domain-specific feedback and reported results.
