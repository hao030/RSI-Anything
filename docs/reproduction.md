# 固定检查点与训练链复现

[返回项目首页](../README.zh-CN.md) · [English](reproduction.en.md)

下列命令均在仓库根目录运行，包含固定检查点评价、训练数据重建和四阶段训练链。

测试集包含 200 道 CodeContests 题目，推理种子为 0、1、2，共执行 600 次评价。每次运行的成功率和各种子成绩由 `summary.json` 汇总。

## 环境与资产

原始环境：Linux、Python 3.10.12、NVIDIA A800 80GB；Python 依赖版本见 `tasks/code_repair/requirements.txt`，原始设备信息见 `tasks/code_repair/config/environment.json`。代码执行器需要可用的 `runuser`、`unshare` 和 `nobody` 用户，保持原实验的隔离执行和 stdout 判题方式。

```bash
python3.10 -m venv .venv-code-repair
. .venv-code-repair/bin/activate
pip install -r tasks/code_repair/requirements.txt
```

安装公共入口后，直接下载带版本号和 SHA256 校验的资产：

```bash
pip install -e .
# 最终检查点、评价输入和 3B 世界模型
rsi assets code_repair --profile evaluate --output assets
# 四阶段训练资料、全部参考检查点和两个基础模型
rsi assets code_repair --profile full --output assets
rsi assets code_repair --profile full --output assets --verify
```

两种配置使用同一目录，可以从 `evaluate` 升级到 `full`；已经校验通过的文件会复用。评价包约 64 MB，训练补充包约 236 MB；3B 基础模型约 6.2 GB，完整配置的两个模型合计约 21.4 GB。解压文件和下载缓存会额外占用空间。详细内容、版本、离线准备方式和许可证见 [资产指南](assets.md) 与 [资产说明](ASSET_CARD.md)。

`assets/manifest.json` 记录已安装文件的大小、SHA256 和来源。训练来源元数据使用可移植标识，原始和发布版训练文件哈希均由配置固定。

已有原始实验目录时，也可生成本机资产副本：

```bash
python tasks/code_repair/export_assets.py --source-root /path/to/experiment-source --output /path/to/portable-assets --mode copy
python tasks/code_repair/reproduce.py verify-assets --assets /path/to/portable-assets
```

`--mode symlink` 可复用本机存储；复制到另一台机器时使用 `--mode copy`。

## 精确输入下重新测试 600 次

```bash
python tasks/code_repair/reproduce.py evaluate \
  --assets assets \
  --gpus 0,1,2,3,4,5,6,7 \
  --inputs recorded \
  --output runs/evaluation
```

该命令重新加载最终 JEPA 世界模型，对每一步的四个修改候选和当前程序重新评分、重新作出选择。全部 600 次决策完成并封存后，执行隐藏测试并汇总成绩。

`recorded` 固定原实验的初始程序、首轮候选、后续已采样候选及公开执行反馈。公开反馈中的执行耗时也属于模型输入，因此需要保留。固定输入缓存包含公开候选和公开执行结果；世界模型评分和决策在本次运行中计算。遇到记录之外的状态时，程序报告缺失状态并停止。

输出：

- `protocol.json`：模型身份、输入模式、方法配置与来源哈希。
- `public/*.jsonl`：本次重新计算的评分和决策。
- `public_seal.json`：全部 600 条决策的封存记录。
- `results.jsonl`：本次隐藏判题结果。
- `summary.json`：实际成功数、各种子成绩及参考决策匹配数。
- `complete.json`：汇总文件校验值。

`successes` 和 `per_seed_successes` 分别记录本次判题的总成功数与各种子成绩；`reference_decisions_matched` 记录与参考决策的匹配数。

## 重建训练数据

```bash
python tasks/code_repair/reproduce.py prepare --assets assets --output runs/prepared
```

该命令从已记录的 TRAIN 执行转移重新构造三阶段训练数据，核对其 SHA256 是否与当前资产版本的固定训练文件一致，并重建完整候选组的批次。公开记录和整套执行质量记录分别路由到对应读出头。

| 阶段 | 训练记录 | 固定配置 |
|---|---:|---|
| 初始 JEPA | 32,920 | seed 0，原生 OPF，1 epoch |
| 执行质量适配 | 7,872 | seed 1，读出头 1 epoch＋联合更新 1 epoch，蒸馏权重 0.25 |
| 扩大训练 | 41,088 | seed 1，联合更新 1 epoch，蒸馏及组排序权重均为 1 |
| 原程序配对更新 | 22,192 | seed 0，1 epoch，配对权重 4，整套预测锚定权重 0，学习率减半 |

训练从记录好的执行转移开始，原始上下文、标签和耗时由资产哈希固定。

## 从头重跑完整训练链

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/code_repair/reproduce.py train \
  --assets assets --stage all --output runs/training
```

顺序执行初始 JEPA、执行质量适配、扩大训练、原程序配对更新。每一阶段以上一阶段本次新训练的检查点为输入。训练配置由开发集选择，并保存在 `tasks/code_repair/config/experiment.json`。

也可从已验证的上一阶段检查点重跑指定阶段：

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/code_repair/reproduce.py train \
  --assets assets --stage incumbent --output runs/incumbent
```

对新训练检查点进行测试：

```bash
python tasks/code_repair/reproduce.py evaluate \
  --assets assets --checkpoint runs/incumbent/joint_checkpoint \
  --inputs fresh --gpus 0,1,2,3,4,5,6,7 --output runs/retrained-evaluation
```

`fresh` 保留同一批初始程序和首轮候选，重新生成后续候选并重新执行公开测试。本次生成的轨迹、执行记录和实际成绩保存在指定输出目录中。

## 方法实现

`tasks/code_repair/src/world.py` 与 `tasks/code_repair/vendor/jepa-anything-core/` 实现原生 JEPA 编码、潜在预测、OPF 和 EMA。`tasks/code_repair/src/semantic_world.py` 实现公开/整套执行读出头以及组排序。`tasks/code_repair/src/losses.py` 实现候选相对原程序的偏好损失。`tasks/code_repair/src/grouping.py` 与 `tasks/code_repair/src/incumbent.py` 保证候选组完整、每批包含四个不同任务。

评价分数为 `P(整套通过) + 0.25 × 预测通过比例`。候选最高分严格超过原程序分数时才替换；平局保留原程序。每轮生成四个候选、进行五次世界模型预测，最多三轮修复。测试期间无优化器更新。

## 检查

```bash
python -m unittest discover -s tasks/code_repair/tests -v
python tasks/code_repair/verify_result.py --assets assets --output runs/evaluation
CUDA_VISIBLE_DEVICES=0 python tasks/code_repair/reproduce.py train \
  --assets assets --stage incumbent --max-steps 3 \
  --no-development --output runs/training-smoke
```

短程训练检查反向传播、检查点保存与重新加载。源码整理时的验证记录见 `tasks/code_repair/config/validation.json`：

| 检查项目 | 验证范围与结果 |
|---|---|
| 评价 | 完成 600 次模型决策及隐藏判题，决策与参考记录匹配 |
| 训练数据 | 三份更新训练数据重建后的哈希与原记录一致 |
| 训练链 | 四个阶段各运行三步，检查点重新加载误差为零 |
| 完整 epoch 重训 | 此次源码整理未执行 |
| 参考检查点几何诊断 | 严格正交及因子独立检查未通过 |

完整训练会保存各阶段的几何诊断结果。

## 文件来源

核心函数来源及哈希见 `tasks/code_repair/config/source_provenance.json`。JEPA-Anything 核心版本为 `c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`，上游许可证保留于 `tasks/code_repair/vendor/LICENSE`。

## 统一 workflow 入口

安装根目录的 `rsi` 后，可直接使用 `configs/code_repair/default.json` 运行四阶段训练及评价，或使用 `configs/code_repair/evaluate.json` 评价已有检查点。配置中的资产路径默认指向仓库根目录的 `assets/`。

```bash
rsi plan --config configs/code_repair/default.json
rsi run --config configs/code_repair/default.json
rsi report --run runs/code-repair
```

统一入口将本次训练的检查点逐阶段传递，并将最终检查点交给评价阶段。本文中的任务命令可用于单独执行训练、数据重建或评价。公共流程说明见 [workflow.md](workflow.md)。
