# 实验室规划

**使用 RSI-Anything，将实验目标转化为可执行的实验操作方案。**

[English](README.md) · [RSI-Anything](../../README.zh-CN.md)

<p>
  <a href="https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning">
    <img src="https://img.shields.io/badge/Hugging%20Face-RSI--Anything-FFB000?style=plastic&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face · RSI-Anything" height="28">
  </a>
</p>

LabHorizon Level 2 要求 Agent 根据实验背景、目标、现有材料、约束和操作库，规划完整的实验步骤。方案需要明确操作顺序、参数，以及中间产物在不同步骤之间的传递关系。

JEPA-Anything 引导实验方案选择。规划 Agent 提出四个候选方案，世界模型预测质量、改进收益、风险和不确定性，进而选择下一步方案。

## RSI 完整流程

RSI 通过多轮更新**规划策略与 JEPA 世界模型**，持续改进实验方案的生成和选择。每一轮都从已保留的模型组合出发，利用任务反馈完成下一次更新。

### 每轮如何改进

完成初始策略和世界模型的训练后，每轮依次进行五个步骤：

1. **生成候选。** 策略读取任务，生成一个贪心方案（`a1`）和三个采样方案（`a2`–`a4`）。贪心方案作为预测改进收益的参照。
2. **预测与选择。** 检查方案结构，在合法候选中综合预测质量与改进收益，扣除风险和不确定性的影响，选择得分最高的合法方案。
3. **获取候选评测反馈。** 在学习任务上，对四个候选方案逐一调用任务评测器，记录操作顺序、参数准确性、中间产物依赖关系等指标及最终规划评分。
4. **更新两个模型。** 用每项任务中实际评分最高的合法候选构造训练样本，从上一轮保留的策略继续训练。世界模型使用累积的“任务—候选方案—评测结果”样本重新训练，形成待验收的新组合。
5. **联合验收与保留。** 在验证任务上评估新组合，比较它与当前组合的规划质量和结构合法性。通过验收的组合成为下一轮的起点；未通过时保留当前组合。

**RSI 的递归机制是模型与经验的逐轮继承。** 保留的策略生成下一轮候选，候选在学习任务上的评测结果继续积累为训练数据。世界模型的方案选择结果用于联合验收，通过验收的模型对成为下一轮改进的起点。

### 冻结模型与最终评测

**验证集与 F200 测试集是独立划分，任务不重叠。** 学习任务用于更新模型，验证集用于逐轮决定保留哪一组策略与世界模型。完成学习与验收后，冻结最终保留的模型组合，再在 **LabHorizon F200** 上统一评测。F200 仅用于模型冻结后的评测与结果报告。

本目录的公开命令对应**训练完成后的冻结推理阶段**：

- `infer.py world` 使用本地规划 adapter 生成四个方案，再由匹配的 JEPA 世界模型选择方案。
- `infer.py select` 对输入的四个候选方案执行相同的世界模型选择。

输出包含选中的方案及预测结果。其中，`forecast.feedback` 是世界模型对评测结果的预测；学习阶段使用任务评测器提供的反馈。

## 环境准备

使用 Python 3.10 或以上版本，从仓库根目录运行：

```bash
python -m pip install -r tasks/laboratory_planning/requirements-world.txt huggingface-hub
export MODEL_DIR="$PWD/models/laboratory_planning"
```

| 推理组件 | 基座或架构 | Adapter 或 checkpoint |
| :--- | :--- | :--- |
| JEPA 候选生成 | [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | 本地 P4 adapter，种子 1401、1402、1403 |
| JEPA 世界模型 | JEPA-Anything | [Hugging Face checkpoint](https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/laboratory_planning/checkpoints) |

下载 JEPA 世界模型 checkpoint：

```bash
hf download Gen-Verse/RSI-Anything --repo-type dataset \
  --include "laboratory_planning/checkpoints/*" \
  --local-dir "$MODEL_DIR/rsi-anything"
```

## 快速开始：选择方案

输入每行包含 `task` 和四个 `candidates`，按 `a1`、`a2`、`a3`、`a4` 排列。每个候选包含 `candidate_id` 和 `completion`，其中 `a1` 为参照方案。格式见[候选示例](examples/candidates.jsonl)。

```bash
python tasks/laboratory_planning/infer.py select \
  --world-dir "$MODEL_DIR/rsi-anything/laboratory_planning/checkpoints/seed1401" \
  --input tasks/laboratory_planning/examples/candidates.jsonl \
  --output outputs/selected_plans.jsonl
```

方案选择在 CPU 上运行。输出包含选中的候选、最终 `completion`、全部候选，以及质量、收益、风险、不确定性和合法性的 `forecast`。添加 `--include-factors` 可输出 JEPA 潜在坐标。

## 任务输入

使用 JSONL，每行一项任务：

| 字段 | 内容 |
| :--- | :--- |
| `id` | 任务标识 |
| `context` | 实验背景 |
| `goal` | 实验目标 |
| `constraints` | 约束列表 |
| `available_inputs` | 描述现有材料的 JSON 编码字符串 |
| `action_pool` | 可用实验操作的 Python 函数定义 |

[任务示例](examples/tasks.jsonl) 展示了输入格式。输出方案以关键字参数调用提供的操作，并通过命名变量传递中间产物。

## 使用本地规划 adapter 生成候选

使用本地 P4 规划 adapter 生成四个候选，再由对应的 JEPA 世界模型选择方案：

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

策略生成一个贪心候选和三个采样候选。JEPA 选择综合考虑预测质量、改进收益、风险、不确定性和方案结构合法性。推理期间，策略模型和世界模型的权重保持冻结。

种子 1402、1403 使用各自对应的 adapter、世界模型和种子参数。

## 代码结构

| 文件 | 作用 |
| :--- | :--- |
| [infer.py](infer.py) | 推理命令行入口 |
| [generator.py](src/lab_inference/generator.py) | 候选方案生成 |
| [world.py](src/lab_inference/world.py) | JEPA 预测与候选选择 |
| [features.py](src/lab_inference/features.py) | 任务文本和方案结构特征 |
| [postprocess.py](src/lab_inference/postprocess.py) | 候选方案格式规范化 |

JEPA-Anything 组件保留 [Apache-2.0](licenses/JEPA-Anything.LICENSE)，LabHorizon 方案解析器保留 [MIT](licenses/LabHorizon-parser.LICENSE)。
