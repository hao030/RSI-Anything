# RSI 相关工作整理（第一部分）

整理日期：2026-10-29。筛选标准：**以高校 / 实验室为主导、有影响力（顶会或广泛引用）、有宣传、代码开源**。每条信息都经过网页检索核对；没核实到的会写明。GitHub star 数查不到（当前环境访问不了 GitHub API），影响力按发表会议、奖项和媒体报道判断。

---

## 我想做的 topic：RSI 的「自评」瓶颈

RSI 能不能持续转起来，取决于模型对自己输出的判断（自评）是否可靠：
- 代码、数学有执行器和标准答案，可以绕开这个问题（Absolute Zero、Darwin Gödel Machine 都依赖外部执行）；
- 一旦换到没有验证器的场景（视觉感知、无人机、具身），改进信号只能来自模型自己：多数投票、自己打分、自问自答、置信度、世界模型预测。**这些信号什么时候可靠、什么时候会把系统带偏，目前缺少系统研究。**

所以下面的工作按「改进信号从哪来」分组整理。

---

## A. 外部验证信号：代码执行 / 基准测试

| 工作 | 机构 | 发表 | 代码 | 宣传 | 一句话 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Absolute Zero** | 清华 LeapLab、北京通用人工智能研究院、Penn State | NeurIPS 2025 | [LeapLabTHU/Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) | 量子位报道；智源社区作者解读 | 零外部数据，模型自己出题、自己解题，用代码执行器验证 |
| **Darwin Gödel Machine** | UBC (Jeff Clune)、Vector Institute、Sakana AI | ICLR 2026 | [jennyzzt/dgm](https://github.com/jennyzzt/dgm) | Sakana 官方博客；英文科技媒体广泛报道 | 智能体改写自己的代码，SWE-bench 从 20% 提到 50% |
| **ADAS** | UBC、Vector Institute (Jeff Clune) | ICLR 2025 | [ShengranHu/ADAS](https://github.com/ShengranHu/ADAS) | NeurIPS 2024 Open-World Agent Workshop 杰出论文 | 元智能体自动设计新的智能体结构 |

## B. 自身一致性 / 自我博弈信号

| 工作 | 机构 | 发表 | 代码 | 宣传 | 一句话 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TTRL** | 清华、上海 AI Lab（周伯文团队） | NeurIPS 2025 | [PRIME-RL/TTRL](https://github.com/PRIME-RL/TTRL) | 量子位报道 | 在无标签测试题上用多数投票当伪标签做 RL；训练后准确率超过伪标签本身 |
| **SPIN** | UCLA (Quanquan Gu) | ICML 2024 | [uclaml/SPIN](https://github.com/uclaml/SPIN) | HF 上公开各轮模型和数据 | 和上一轮的自己博弈，只用 SFT 数据就能持续提升 |
| **R-Zero** | 腾讯 AI Seattle Lab，作者含华盛顿大学圣路易斯分校（企业参与） | ICLR 2026 | [Chengsong-Huang/R-Zero](https://github.com/Chengsong-Huang/R-Zero) | — | Challenger 出难题、Solver 解题，奖励来自自身一致性 |

## C. 多模态自我进化（你拉进仓库的那条线）

| 工作 | 机构 | 发表 | 代码 | 宣传 | 一句话 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VisPlay** | UIUC 等 | CVPR 2026 | [项目主页](https://bruno686.github.io/VisPlay/) | 有项目主页；没找到中文媒体报道 | 同一个 VLM 既出视觉题又答题，只用无标签图像 |
| **Vision-Zero** | Duke、NUS、UMD、Adobe | ICLR 2026 | [wangqinsi1/Vision-Zero](https://github.com/wangqinsi1/Vision-Zero) | 没找到中文媒体报道 | 用「谁是卧底」式视觉游戏自我博弈，交替做 RLVR 防止停滞 |
| **Vision-SR1** | 第一作者来自 UMD，作者含腾讯 AI Lab（企业参与） | ICLR 2026 | [zli12321/Vision-SR1](https://github.com/zli12321/Vision-SR1) | — | 推理拆成「看」和「想」两步，模型自己验证、自己给奖励 |
| **Ranked from Within** | 澳大利亚国立大学、悉尼大学 | ICML 2025 | **没找到代码**（论文内也没有链接） | ICML 有演讲视频 | 不用标签，靠 softmax 不确定性给 47 个多模态模型排序 |
| RISE | 阿里高德（大厂，仅作参照） | arXiv 2605.20914 | [AMAP-ML/RISE](https://github.com/AMAP-ML/RISE) | — | 指出自我进化中出题质量会下降、题型会收窄，加质量监督 |

## D. 经验 / 记忆信号（免训练）

| 工作 | 机构 | 发表 | 代码 | 宣传 | 一句话 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reflexion** | Northeastern、MIT、Princeton（机构信息来自二手资料） | NeurIPS 2023 | [noahshinn/reflexion](https://github.com/noahshinn/reflexion) | 智能体领域的奠基工作之一 | 失败后用文字反思，存进记忆，下次重试时参考 |
| **Self-Refine** | CMU 主导，含 AI2、UW 等 | NeurIPS 2023 | [madaan/self-refine](https://github.com/madaan/self-refine) | [项目主页](https://selfrefine.info/) | 同一个模型生成、自评、修改，循环迭代 |
| **ExpeL** | 清华（黄高组） | AAAI 2024 | [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL) | Paper Digest 评为 AAAI 2024 最具影响力论文第 4 名 | 从训练任务的成败中提炼经验规则，不更新模型参数 |
| **Steve-Evolving** | 作者含中科大（据代码账号推测） | arXiv 2603.13131 | [xzw-ustc/MC-MineEvolve](https://github.com/xzw-ustc/MC-MineEvolve) | — | Minecraft 长程任务，细粒度诊断失败原因 + 双轨知识沉淀；可能和 NeurIPS 2026 的 MineEvolve 是同一工作，待核 |

## E. 世界模型预测信号

| 工作 | 机构 | 发表 | 代码 | 宣传 | 一句话 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RSI-Anything** | PhAI Labs、港中文、澳门大学、上交、复旦、牛津、斯坦福、北大 | arXiv 2610.13912 | 本仓库 | 机器之心发布（机构投稿稿件） | 世界模型先预测后果，决定哪些候选值得真实评估；跨 5 个领域 |
| **Dream-RSI** | UMD 牵头，含 Google / DeepMind（企业参与） | arXiv 2609.14858 | [zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI) | 36 氪等报道 | 用执行日志构建回放模拟器，在「梦」里改进探索策略 |

## F. 综述（入门用）

| 综述 | 机构 | 发表 | 资源 |
| :--- | :--- | :--- | :--- |
| A Survey of Self-Evolving Agents | Princeton (Mengdi Wang)、清华、CMU、上交等 | TMLR 2026 | [CharlesQ9/Self-Evolving-Agents](https://github.com/CharlesQ9/Self-Evolving-Agents) |
| Self-Improvement in Multimodal LLMs: A Survey | UT Dallas (Yapeng Tian) 等 | Findings of EMNLP 2025 | 已在 `papers/` |
| Self-evolving Embodied AI | 清华（朱文武组） | arXiv 2602.04411 | 具身方向，提出记忆、任务、环境预测、本体适应、模型进化五个模块 |

---

## 建议先精读并复现的 2–3 篇

按「好论文有一两篇就够，但一定要复现」的思路，结合上面的 topic：

1. **TTRL**：流程最简单（采样 → 投票 → RL），代码开源，而且「多数投票什么时候失效」正好是 topic 的核心问题。复现之后可以直接拿来做自评信号的对照实验。
2. **VisPlay**：多模态、CVPR 2026、有开源代码，和组里做 VLM / 视频的方向最接近；后续可以换到无人机视频数据上。
3. **Ranked from Within**（备选）：没有官方代码，但方法只用推理阶段的不确定性，自己实现一遍成本很低，可以作为「无标签自评」的基线。

Darwin Gödel Machine 和 ADAS 影响力大，但大量调用闭源 API，复现费用高，适合读懂思路，不建议先复现。

---

## 没核实到的

- 各仓库的 star 数和最近更新时间；
- Reflexion 的作者机构（只找到二手资料）；
- Steve-Evolving 和 MineEvolve 是不是同一个工作；
- VisPlay、Vision-Zero 是否有中文媒体报道（检索没找到，不代表没有）。
