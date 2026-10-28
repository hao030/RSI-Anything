# RSI-Anything experiment assets

## Contents and use

The Code Repair asset release supplies fixed public candidate/execution records,
200 CodeContests judging tasks, reference decisions, four trained world-model
checkpoints, development records, and training transitions. Use `evaluate` for
recorded-input evaluation or `full` for the four-stage training workflow and fresh
candidate generation. Scores are computed by the evaluator in the current run.

The GUI pipeline collects its own browser transitions and trains its checkpoints.
Its task HTML, split definitions and reported episode endpoints are in the source
repository; its two base models use the same fixed upstream revisions.

## Provenance and file identities

The source repository includes a versioned catalog of archive and file SHA256
hashes. Changed metadata also retains `original_sha256`. Training provenance uses
portable `experiment-source/` identifiers. Adapter configuration names the
upstream model; nested checkpoint manifests are recomputed. Tensor files, model
inputs, labels, sample order, execution timings and evaluator settings are retained.
The training configuration records both original and portable file hashes.

- World backbone: Qwen/Qwen2.5-Coder-3B-Instruct,
  revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Policy: Gen-Verse/ReasonFlux-Coder-7B,
  revision `6860a2fd222bd2e18e8a5b62a9c0887d2ba70c36`.
- Dataset: [CodeContests](https://github.com/google-deepmind/code_contests),
  introduced by Li et al., *Competition-Level Code Generation with AlphaCode*,
  Science (2022). The released subset preserves task IDs and judging examples.
- Project: [RSI-Anything](https://github.com/Gen-Verse/RSI-Anything).

## Licenses and attribution

Project-owned source code is Apache-2.0. Built with Qwen.

Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT,
Copyright (c) Alibaba Cloud. All Rights Reserved.
The included `Qwen-RESEARCH-LICENSE.txt` covers the world backbone and its adapted
weights; their use is for noncommercial research and evaluation under those terms.
The adapters were trained by RSI-Anything contributors for execution prediction.
Base models are downloaded from their upstream repositories together with their
model cards and available license files. ReasonFlux's model card identifies MIT.

CodeContests non-code material is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with source attribution
listed in the [upstream dataset documentation](https://github.com/google-deepmind/code_contests#license).
RSI-Anything contributors provide the derived transition records and decision
annotations under CC BY 4.0, retaining those upstream notices. Changes include
subset selection, program generation and execution, and transition annotation.
