# Suzuki 反应优化

使用已校准的 ReactionT5–JEPA 世界模型预测和实测产率反馈，推荐下一组 Suzuki–Miyaura 反应条件。

[English](README.md) · [RSI-Anything](../../README.zh-CN.md)

<p>
  <a href="https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/reaction_optimization">
    <img src="https://img.shields.io/badge/Hugging%20Face-RSI--Anything-FFB000?style=plastic&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face · RSI-Anything" height="28">
  </a>
</p>

## 结果

在 Flow LC–MS 筛选数据上，以逐步揭示实测产率的方式进行离线顺序评测。每次运行从 8 个已测条件开始，总预算为 40 个不同条件，包含初始 8 个。

| 指标 | 结果 |
| :--- | ---: |
| 候选条件数 | 321 |
| 运行次数，种子 4001–4100 | 100 |
| 第 10 次观测时的平均最优实测产率 | **96.15%** |
| 第 40 次观测时的平均最优实测产率 | **96.15%** |
| 在第 10 次观测以内找到候选池最高产率的运行 | **100 / 100** |
| 首次找到最高产率的平均观测次数 | **9.94** |

96.15% 表示反应的实测产率。这一结果对应固定反应体系及其筛选条件池。完整设置、校准参数和文件身份见[评测记录](results/evaluation.json)，发布代码的验证结果见[发布校验](results/validation.json)。

## 安装与快速运行

使用 Python 3.10 或更高版本，在仓库根目录运行：

```bash
python -m pip install -r tasks/reaction_optimization/requirements.txt
python tasks/reaction_optimization/infer.py \
  --observations tasks/reaction_optimization/examples/observations.json \
  --output runs/suzuki/next_condition.json
```

示例中的产率为**演示用的合成数值**，使用时替换为自己的测量值。省略 `--output` 可直接输出 JSON；指定输出文件时，程序创建新文件，已有同名文件会报错。

发布的预测表包含世界模型对这个候选池的固定预测及任务校准结果，安装 NumPy 和 SciPy 后即可在 CPU 上选择条件。

## 输入与反馈循环

将已测条件写成 JSON 列表：

```json
[
  {"candidate_index": 0, "yield": 42.0},
  {"candidate_index": 1, "yield": 55.0}
]
```

| 字段 | 含义 |
| :--- | :--- |
| `candidate_index` | 0–320 的整数，对应发布候选目录中的条件编号。 |
| `yield` | 0–100 的产率百分数，例如 `55.0`。 |

输入至少包含一个观测，报告中的评测设置使用 8 个初始观测。条件编号必须唯一。测量推荐条件后，将它的 `candidate_index` 和实测产率追加到列表，再运行推理。反馈更新期望改进所用的当前最高产率，并排除已经测量的条件；循环中的预测表保持固定。

[候选目录与预测表](assets/suzuki_prior.json) 对应亲电试剂 **1c, 6-OTf-Q**、亲核试剂 **2b, Boronic Ester** 和催化剂 **Pd(OAc)2**，包含 321 个配体、碱和溶剂组合。每条记录给出完整条件名称。`original_table_index` 用于映射到筛选后的 5,030 行原始表；提交观测时使用 `candidate_index`。

## 输出

| 字段 | 含义 |
| :--- | :--- |
| `candidate_index` | 下一组尚未测量的推荐条件。 |
| `original_table_index` | 对应筛选表中的行号。 |
| `conditions` | 亲电试剂、亲核试剂、催化剂、配体、碱、溶剂名称。 |
| `predicted_yield_percent` | 任务适配后的固定平均预测产率。 |
| `calibrated_std_percent_points` | 已校准的预测标准差，单位为产率百分点。 |
| `expected_improvement` | 期望改进分数，单位为产率百分点。 |
| `observations_used` | 输入的测量次数。 |
| `best_observed_yield_percent` | 输入观测中的最高产率。 |
| `model_id` | 当前发布的 Suzuki 预测表标识。 |

[推荐示例](examples/recommendation.json) 由上述合成观测生成。每次推荐一个条件；分数相同时选择编号更小的条件。全部候选都已测量时，程序给出明确提示。

## Python 调用

在仓库根目录运行：

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

后续推荐可以复用 `prior`。`load_prior(path)` 与命令行 `--prior path` 可加载同一格式的预测表。

## 校验与文件

```bash
python tasks/reaction_optimization/scripts/verify_assets.py
python -m unittest discover -s tasks/reaction_optimization/tests -v
```

| 文件 | 用途 |
| :--- | :--- |
| [infer.py](infer.py) | 可移植命令行入口。 |
| [prior.py](src/suzuki_inference/prior.py) | 预测表和条件目录读取、校验。 |
| [selector.py](src/suzuki_inference/selector.py) | 期望改进与顺序推荐。 |
| [cli.py](src/suzuki_inference/cli.py) | 观测输入与 JSON 输出。 |
| [suzuki_prior.json](assets/suzuki_prior.json) | 固定校准预测与候选条件。 |
| [manifest.json](assets/manifest.json) | 预测文件校验值与来源身份。 |
| [evaluation.json](results/evaluation.json) | 当前模型的实测结果与评测设置。 |

项目自有代码遵循仓库的 [Apache-2.0 许可证](../../LICENSE)。ReactionT5 与 JEPA 的组件声明保留在 [licenses](licenses/) 中。
