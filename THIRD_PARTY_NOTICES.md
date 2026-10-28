# Third-party components, models and data

Project-owned RSI-Anything code is licensed under [Apache-2.0](LICENSE).
Built with Qwen.

| Component | Terms and included notice |
|---|---|
| JEPA-Anything core | Apache-2.0; `tasks/code_repair/vendor/LICENSE` and `tasks/gui/licenses/JEPA-Core.LICENSE` |
| MiniWoB++ task resources | MIT; `tasks/gui/vendor/miniwob-plusplus/LICENSE`; original dependency headers are retained |
| Qwen2.5-Coder-3B-Instruct | Qwen Research License; `tasks/gui/licenses/Qwen2.5-Coder-3B-Instruct.LICENSE`; base revision pinned in the asset catalog |
| ReasonFlux-Coder-7B | Upstream model card identifies MIT; retain its model card and underlying base-model notices with downloaded weights |
| CodeContests | Upstream code: Apache-2.0; non-code materials: CC BY 4.0, with the source-specific notices listed upstream |
| ReactionT5–JEPA Suzuki adaptation | ReactionT5: MIT, `tasks/reaction_optimization/licenses/ReactionT5v2-LICENSE.txt`; JEPA: Apache-2.0, `tasks/reaction_optimization/licenses/JEPA-Anything-LICENSE` |
| Laboratory JEPA-Anything inference | Apache-2.0; `tasks/laboratory_planning/licenses/JEPA-Anything.LICENSE` |
| LabHorizon plan parser | MIT; `tasks/laboratory_planning/licenses/LabHorizon-parser.LICENSE` |

Model and adapter releases include their own license materials and base-model
identity. The Qwen 3B model's research license applies to its use and derivatives;
commercial use requires a separate grant from its licensor.

## Sources

- [JEPA-Anything](https://github.com/Gen-Verse/JEPA-Anything)
- [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus)
- [Qwen model license](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/488639f1ff808d1d3d0ba301aef8c11461451ec5/LICENSE)
- [ReasonFlux model card](https://huggingface.co/Gen-Verse/ReasonFlux-Coder-7B/blob/6860a2fd222bd2e18e8a5b62a9c0887d2ba70c36/README.md)
- [CodeContests data, attribution and license](https://github.com/google-deepmind/code_contests#license)

The GUI task also supplies a [component inventory](tasks/gui/THIRD_PARTY_NOTICES.md).
