# RSI-Anything：GUI 实验

<p>
  <a href="https://huggingface.co/datasets/Gen-Verse/RSI-Anything/tree/main/gui_interaction">
    <img src="https://img.shields.io/badge/Hugging%20Face-RSI--Anything-FFB000?style=plastic&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face · RSI-Anything" height="28">
  </a>
</p>

完整英文文档见 [README.md](README.md)。GUI 工作流涵盖采集、JEPA 世界模型训练、策略适配、控制器选择、交叉测试、条件确认、审计及统计。

## 实验结果

Agent 读取任务指令和可见 DOM，在 Chromium 中点击指定复选框并提交。**严格成功**要求全部复选框状态符合指令且完成提交，限时 120 秒、最多八次动作（包含提交）。

三个保留版本在同一组 100 个 RSI 测试实例上评价：

| 指标 | Baseline | Adapted checkpoint | RSI-Anything |
|---|---:|---:|---:|
| 选定控制器成功数 | 51/100 | 79/100 | **99/100** |
| 世界模型控制成功数 | 49/100 | 96/100 | **98/100** |
| 选定控制器平均动作数／回合 | 5.30 | 2.93 | 3.21 |
| 选定控制器重复点击总数 | 237 | 2 | **0** |
| 选定控制器错误提交总数 | 9 | 21 | **1** |

**Baseline → Adapted checkpoint → RSI-Anything** 依次对应更新前的策略／世界模型组合、一次更新后保留的组合、两次更新后保留的最终组合。每次更新采集浏览器经验，用成功动作与回放数据适配策略，用真实转移与专家后续完成记录更新世界模型，再通过开发集选择保留版本；测试时模型参数冻结。

**选定控制器**是开发集选出的执行规则，本实验选中 `alpha=0`，按策略排序执行。**世界模型控制**用世界模型预测值，对同一策略提供的候选动作（包含提交）排序。表中的两行成功数对应每个版本的两种测试方式。

最终选定控制器相对基线每 100 个任务多完成 **48 个**。平均动作数包含成功与失败回合；重复点击和错误提交为 100 个回合的总数。中间版本的 2.93 次平均动作，需要与其 21 次错误提交一起理解。

## 测试集与世界模型结果

| 测试集合 | 实例数 | 对应结果 |
|---|---:|---|
| RSI test | 100 | 三版本完整 workflow：51% → 79% → 99%；世界模型控制 49% → 96% → 98% |
| World test | 100 | 固定基础策略候选，三个 OPF 模型更新前 52%／40%／44%，更新后 95%／95%／96% |
| Confirmation | 630 | 同三个固定 OPF 模型逐一进行更新前后配对评价，平均 43.81% → 97.04% |

三个集合使用互不重叠的实例种子。确认集平均提升 53.23 个百分点，按任务配对 bootstrap 的 95% 置信区间为 [50.26, 56.30] 个百分点。World test 中的 61% 策略基线与 RSI test 中的 51% 分别来自各自测试集。

交叉评价将旧／新策略与旧／新世界模型组成四种组合；每次更新的“旧”均指该次更新之前保留的版本。全部指标和文件标识见 [RESULTS.md](docs/RESULTS.md)。

## 统一 workflow

从仓库根目录安装公共 `rsi` 入口，按[英文文档](README.md#run-through-the-shared-workflow)安装根目录的 `.venv-gui` 任务环境，然后运行：

```bash
rsi plan --config configs/gui/default.json
rsi run --config configs/gui/default.json
rsi report --run runs/gui
```

公共调度器记录各阶段状态并汇总结果。下方任务原生命令适合单独执行某一步。

## 快速验证

以下命令在集成仓库的 `tasks/gui/` 中执行；若使用独立 GUI 源码包，则在其根目录执行。浏览器验证无需 GPU 或模型：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[browser,test]'
python -m playwright install chromium
rsi-anything-gui --config configs/full.json smoke --episodes 4
```

此命令通过规则控制器执行 Chromium 点击与提交，并保存快速验证报告。

## 测试与统计复算

完整测试使用 PyTorch、NumPy 和 SciPy 等依赖。按英文文档准备完整环境，或参照仓库 GUI CI 配置建立 CPU 检查环境后，在 GUI 目录执行：

```bash
RSI_TEST_BROWSER=1 python -m pytest
python scripts/reproduce_reported_statistics.py
python scripts/verify_release.py
```

`RSI_TEST_BROWSER=1` 启用已安装 Chromium 的浏览器测试。统计脚本从随附结果记录复算 13 项配对比较。

## 完整实验

按英文文档安装 Python 3.10、CUDA 12.1 对应的 PyTorch 和固定依赖。修改配置中的新实验目录、物理 GPU 编号和可选模型目录后：

```bash
rsi-anything-gui --config configs/full.json download-models
rsi-anything-gui --config configs/full.json doctor --gpu
rsi-anything-gui --config configs/full.json freeze
rsi-anything-gui --config configs/full.json preflight browser
rsi-anything-gui --config configs/full.json preflight world
rsi-anything-gui --config configs/full.json run
```

默认配置用三张独立 GPU，将五组世界实验串行执行；并行配置用六张。实验硬件为 A800 80 GB。

## 源码与资料

包内包含源代码、官方任务页面、JEPA 核心、许可证、固定模型输入哈希、脱敏统计材料、测试和文档。源码来源与验证记录见 [PROVENANCE.md](docs/PROVENANCE.md)、[VERIFICATION.md](docs/VERIFICATION.md)。
