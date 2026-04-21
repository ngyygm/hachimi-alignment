# Hachimi-Alignment 研究批判性评估与后续方案

> **文档目的**：对当前研究（论文、数据集、实验代码）进行全面批判性审视，明确现存问题，并为后续研究者提供详细的实验设计与论文改进方案。
>
> **当前状态**：论文已完成 6 轮自动审稿循环，最终评分 Workshop 8.5/10、Findings 7/10。目标投稿 ACL 2025。
>
> **撰写时间**：2026-04-21

---

## 目录

1. [研究概述](#1-研究概述)
2. [论文层面的问题](#2-论文层面的问题)
3. [数据集层面的问题](#3-数据集层面的问题)
4. [实验代码层面的问题](#4-实验代码层面的问题)
5. [实验设计层面的问题](#5-实验设计层面的问题)
6. [后续实验设计方案](#6-后续实验设计方案)
7. [论文改写方案](#7-论文改写方案)
8. [优先级与执行路线图](#8-优先级与执行路线图)

---

## 1. 研究概述

### 1.1 核心问题

本研究探究 CLAP（Contrastive Language-Audio Pretraining）模型在音频-文本对齐时，文本的**语义内容**是否真正参与了对齐计算。

### 1.2 研究方法

利用中文互联网文化中的"哈基米歌词"现象作为自然探针：哈基米歌曲将原曲的有意义歌词替换为无意义音节（如"哈基米"、"曼波"），但保留旋律、配器和人声音色。这构成了一个天然的对照实验——语义被移除，但声学属性保持不变。

### 1.3 核心发现

- **释义不变性**（C0 ≈ C8）：LAION CLAP 两个变体中，保留语义但更换词语的释义文本与原文对齐度无显著差异（d = -0.02, p = 0.82; d = 0.14, p = 0.08）
- **哈基米优势**（C1 > C0）：所有三个模型中，无意义哈基米文本的对齐度均显著高于原始歌词（d = -0.24 至 -0.37）
- **模型依赖性**：MS-CLAP 能区分释义与原文（d = 0.45），表明释义敏感性因模型而异
- **同音字控制**：保留语音但更换字形会降低对齐度（d = -0.27），排除了纯语音学解释

### 1.4 现有实验清单

| 脚本 | 功能 | 输出 |
|------|------|------|
| `1_generate_paraphrases.py` | 通过 MiniMax API 生成释义 | `paraphrases.json` |
| `2_compute_alignment.py` | 计算 LAION/MS-CLAP 对齐度 | `cleaned_results.json` |
| `3_match_segments.py` | 色度互相关时间对齐 | `segment_match_aligned.json` |
| `4_export_segments.py` | 导出匹配音频片段 | WAV 文件 |
| `5_matched_alignment.py` | 匹配片段上的对齐度 | `matched_segment_results.json` |
| `6_segment_analysis.py` | 对比实验（时长、质量等） | `comparison_experiments_results.json` |
| `7_generate_figures.py` | 生成论文图表 | PDF/PNG |
| `homophone_experiment.py` | 同音字语音控制实验 | `homophone_results.json` |
| `msclap_truncation_control.py` | MS-CLAP 截断长度控制 | `msclap_truncation_results.json` |
| `fused_clap_experiment.py` | LAION-Fused 变体实验 | `fused_clap_results.json` |
| `retrieval_experiment.py` | 条件排名检索指标 | `retrieval_results.json` |

---

## 2. 论文层面的问题

### 2.1 核心论证逻辑缺陷

**问题 2.1.1：因果声明与观察性证据不匹配**

论文标题"When Meaning Dissolves"暗示了一个因果叙事（语义消失→对齐不变），但实际上所有证据都是观察性的。C0 vs C1 的对比同时改变了语义、音节结构、韵律、分词方式等多个变量，无法从中得出"语义不重要"的因果结论。

虽然论文 Discussion 部分已承认这一点（"Observational findings, not causal claims"），但标题和 Abstract 的措辞仍然暗示因果关系。例如 Abstract 中"whether semantic content in text contributes to this alignment"的表述预设了一个可以被回答的因果问题，但实验设计无法回答它。

> **建议**：标题应改为更审慎的表述，如 "What Do CLAP Models Respond To? A Case Study with Chinese Lyric Perturbations"。Abstract 应明确标注这是观察性案例研究。

**问题 2.1.2：C0 ≈ C8 的论证力度被高估**

论文将"释义与原文对齐度无差异"作为"语义对对齐无贡献"的最强证据。但这一推理有如下漏洞：

1. **空效应不等于零效应**：p = 0.82 不代表 C0 和 C8 完全相同，只是当前样本量（N = 165）下无法检测到差异。需要等效性检验（TOST）来正面论证"两者等价"。
2. **释义质量未经人工验证**：LLM 生成的释义可能在"可唱性"、词汇罕见度、节奏感等维度与原文存在系统性差异，这些差异可能恰好与语义变化抵消。
3. **释义使用了不同的 LLM**：论文声明使用 Qwen2.5-7B，但代码 `1_generate_paraphrases.py` 实际调用的是 MiniMax API（MiniMax-M2.7 模型），这是一个不一致。

> **建议**：(1) 添加 TOST 等效性检验（建议 δ = 0.1）；(2) 完成人工评估至少 50 首歌的释义质量；(3) 统一论文与代码中的模型描述。

**问题 2.1.3：效应量偏小且绝对对齐度低**

所有核心效应的绝对对齐度非常低（余弦相似度 ≈ 0.06-0.08），且效应量属于小-中等（|d| = 0.24-0.37）。论文已承认这一点，但对"低信号区间下的结论可靠性"讨论不够充分。

在余弦相似度接近 0 的区间，微小的数值波动可能被统计检验捕获为"显著"，但实际意义存疑。例如 C0 = 0.062 vs C1 = 0.084 的差异仅为 0.022，这是否具有实践意义？

> **建议**：增加一节讨论"实践显著性"，解释为什么在低信号区间中这些相对排序仍然有意义。可以参考 MIR 领域的标准来定义"有意义的对齐差异"。

### 2.2 方法描述问题

**问题 2.2.1：释义生成流程描述不一致**

论文 Section 3.3 写的是 "Qwen2.5-7B via local inference"，但实际代码 `1_generate_paraphrases.py` 通过 `curl` 调用的是 MiniMax 的远程 API（模型为 MiniMax-M2.7）。这是一个事实性错误，如果审稿人检查代码会直接发现。

此外，代码中歌词截断为 200 字符（第 79 行 `lyrics[:200]`），而论文写的是 500 字符。这意味着释义实际上只基于歌词的前 200 个字符生成，远少于论文描述。

> **必须修正**：论文中释义模型和截断长度的描述必须与代码一致。

**问题 2.2.2：Fused CLAP 模型初始化存疑**

`fused_clap_experiment.py` 中 CLAP-Fused 模型的加载代码为：
```python
clap_fused = CLAP_Module(enable_fusion=True)
# 缺少: clap_fused.load_ckpt()
# 缺少: clap_fused = clap_fused.cuda().eval()
```

与 `2_compute_alignment.py` 中标准 CLAP 的加载对比：
```python
clap = CLAP_Module(enable_fusion=False)
clap.load_ckpt()       # 加载预训练权重
clap = clap.cuda().eval()  # GPU + 推理模式
```

Fused 模型缺少 `load_ckpt()` 调用和 `.cuda().eval()` 设置。这意味着**Fused CLAP 的结果可能来自未加载预训练权重的模型或 CPU 模式下的计算**。论文中基于 Fused 模型的所有结论（包括 "fused variant shows largest hachimi advantage, d = -0.37"）的可靠性需要重新验证。

> **严重问题**：必须修复代码并重新运行 Fused CLAP 实验，确认结果是否改变。

**问题 2.2.3：HuBERT+BERT CCA 基线在论文中提及但未报告结果**

论文 Section 3.4 描述了 HuBERT+BERT via CCA 作为 "non-jointly-trained baseline"，但在实验结果部分完全没有报告该基线的结果。如果实验已完成但结果不佳而选择不报告，这属于选择性报告偏差。

> **建议**：要么报告 HuBERT+BERT 结果（即使为负面结果），要么从方法部分删除该描述。

### 2.3 统计分析问题

**问题 2.3.1：多重比较校正不完整**

论文报告了对 10 个条件的 Bonferroni 校正（表 1），但以下比较缺少校正：

- 跨模型比较（3 个模型 × 多个条件对）
- 声学特征与对齐度的相关分析（虽然报告了 57 次 Bonferroni 校正，但仅选择性报告了显著结果）
- 扰动实验中的 6 个比较（3 种扰动 × 2 种文本条件）

**问题 2.3.2：Bootstrap CI 的种子固定问题**

所有实验脚本在开头设置 `np.random.seed(42)`，这意味着 bootstrap 重采样也使用了固定种子。虽然有利于可复现性，但 10,000 次 bootstrap 的 CI 实际上是确定性的，不再是真正的随机估计。更严格的做法是对主分析使用固定种子，对 bootstrap CI 使用独立的随机流。

**问题 2.3.3：缺少效应量的置信区间**

论文报告了 Cohen's d 的点估计，但未报告 d 的置信区间。对于 N = 166 的样本量，d = -0.24 的 95% CI 大约为 [-0.39, -0.09]，跨越了"小效应"和"接近零"的边界。报告 CI 能让读者更好地评估效应的不确定性。

> **建议**：为所有关键 Cohen's d 值添加非中心 t 分布的 95% CI。

### 2.4 写作与呈现问题

**问题 2.4.1：论文结构头重脚轻**

Section 4 (Experiments) 包含了 7 个子节（4.1-4.7），占论文主体的 ~70%，而 Discussion 和 Conclusion 相对单薄。大量实验结果堆砌导致读者难以抓住主线。

> **建议**：将部分实验（同音字、截断控制、声学分析）移入附录，主文聚焦 3 个核心实验。

**问题 2.4.2：图表与文字重复**

表 1 和图 1 呈现了几乎相同的信息（各条件的对齐度），表 2 和图 3 也有类似重复。ACL 页数限制下，这浪费了宝贵的空间。

> **建议**：每组结果选择表或图其一，释放空间给讨论和分析。

**问题 2.4.3：相关工作缺少直接竞争对手**

Related Work 覆盖了 CLIP 探测、非语义嵌入等方向，但缺少对以下直接相关工作的讨论：

- 音乐信息检索中文本查询的有效性研究
- 其他语言下 CLAP 的鲁棒性分析
- 中文 NLP 中的分词/音节效应对下游任务的影响

---

## 3. 数据集层面的问题

### 3.1 数据集规模与代表性

**问题 3.1.1：样本量偏小（N = 166）**

166 首歌虽然是目前已知的最大哈基米配对数据集，但对于声称发现"CLAP 模型的普遍行为模式"来说，统计效能有限。Bootstrap CI 对某些条件跨越了均值的 30%（如 LAION C1: [0.071, 0.097]），说明估计精度不高。

**统计效能分析**：对于 d = 0.24（C0 vs C1），N = 166 给出的统计效能约为 0.78（假设 α = 0.05，双侧检验），勉强达到通常要求的 0.80 门槛。对于更小的效应（如 d = 0.14，Fused C0 vs C8），效能仅约 0.35，远不足以下结论。

> **后续建议**：将数据集扩展至 500+ 首歌，或明确报告各比较的事后统计效能。

**问题 3.1.2：数据来源单一**

所有歌曲来自中文互联网文化中的哈基米现象，这意味着：

- **语言限制**：仅覆盖中文，无法推广到英文或其他语言
- **体裁限制**：多为流行/动漫歌曲，不包括古典、摇滚、嘻哈等体裁
- **文化限制**：哈基米是特定互联网亚文化现象，其音节模式（"哈基米"、"曼波"等）可能有独特的声学-文本关联

> **后续建议**：收集至少 2 种其他语言（日语动漫翻唱、英语 misheard lyrics）的类似数据进行跨语言验证。

### 3.2 数据质量

**问题 3.2.1：文本条件未经独立验证**

10 个文本条件（C0-C8）的生成过程在代码中实现，但缺少自动化验证脚本来检查：

- C2（字符打乱）是否真的在词内打乱而非跨词？
- C5（随机音节）生成的是否为合法的中文音节？
- C6/C6b（语义取反）的取反操作是否语法正确？
- C8（释义）的语义相似度分布是否有异常值？

**问题 3.2.2：音频质量不一致**

数据集中的 MP3 文件为可变比特率（VBR），不同歌曲的编码质量可能不一致。此外，哈基米版本多为网络录制，音质参差不齐，可能引入与语义无关的声学差异。

**问题 3.2.3：配对验证缺失**

当前只通过文件名（同一目录下的 `hachimi-*.mp3` 和 `raw-*.mp3`）进行配对，没有自动化验证配对的正确性。如果某个目录下存在多个 hachimi 或 raw 文件，`get_audio_path()` 只取第一个，可能导致错误配对。

### 3.3 数据可用性

**问题 3.3.1：音频版权问题**

论文附录提到 "Audio files are subject to copyright"，HuggingFace 上的数据集页面需要确认是否有明确的使用许可。如果音频无法公开，研究的可复现性大打折扣。

**问题 3.3.2：`data/` 目录为空**

仓库中 `data/` 目录只包含一个指向 `../matched_segments` 的符号链接，而 `matched_segments` 目录在仓库中不存在。完整的数据需要从 HuggingFace 下载。README 中的下载说明应当更加清晰。

---

## 4. 实验代码层面的问题

### 4.1 代码 Bug

**Bug 4.1.1：路径构造类型错误（影响 6 个脚本）**

以下脚本使用 `os.path.dirname()` 返回 `str`，然后对 `str` 使用 `/` 运算符，这在 Python 中会抛出 `TypeError`：

| 脚本 | 行号 | 代码 |
|------|------|------|
| `2_compute_alignment.py` | 24-26 | `PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)` → `DATA_DIR = PROJECT_ROOT / "data" / ...` |
| `3_match_segments.py` | 19-21 | 同上 |
| `4_export_segments.py` | 19-21 | 同上 |
| `5_matched_alignment.py` | 23-25 | 同上 |
| `6_segment_analysis.py` | 25-27 | 同上 |
| `7_generate_figures.py` | 31-33 | 同上 |

而 `homophone_experiment.py`、`fused_clap_experiment.py`、`msclap_truncation_control.py` 使用了 `Path.parent`，不存在此问题。

**这意味着有路径问题的 6 个脚本在当前代码状态下无法运行**。已有的结果文件可能是在修复此 Bug 前（或使用不同版本的代码）生成的，代码与结果的对应关系存疑。

> **修复**：统一使用 `Path(__file__).resolve().parent.parent` 替代 `os.path.dirname()`。

**Bug 4.1.2：Fused CLAP 未加载权重（上文 2.2.2 已详述）**

`fused_clap_experiment.py:53` 缺少 `load_ckpt()` 和 `.cuda().eval()` 调用。

**Bug 4.1.3：释义截断长度不一致**

`1_generate_paraphrases.py:79` 截断为 200 字符：
```python
lyrics = conditions[name]["C0_orig_lyrics"][:200]
```
而论文 Section 3.3 声称 500 字符。这意味着释义只覆盖了歌词的前 200 个字符。

### 4.2 代码质量问题

**问题 4.2.1：大量函数重复定义**

以下函数在 5 个以上脚本中被完整复制：

| 函数 | 出现次数 | 行数/次 |
|------|----------|---------|
| `cos_sim()` | 5 | 4 |
| `bootstrap_ci()` | 5 | 6 |
| `clap_text_embed()` | 3 | 10 |
| `get_audio_path()` | 4 | 7 |

这不仅增加维护成本，还导致了实际不一致：`2_compute_alignment.py` 中的 `clap_text_embed()` 使用 batch_size=16，而 `homophone_experiment.py` 中也是 16，但 `msclap_truncation_control.py` 中的 `msclap_text_embed()` 使用 batch_size=1。

> **建议**：提取到 `scripts/utils.py`，所有脚本统一导入。

**问题 4.2.2：错误处理过于宽松**

多个脚本在嵌入计算失败时填充零向量：
```python
except Exception as e:
    print(f"  Text embed error: {e}, filling zeros for {len(batch)} texts")
    embs.append(np.zeros((len(batch), 512)))
```

零向量的余弦相似度为 0（经过归一化后），会系统性地拉低平均对齐度。如果某些歌曲的嵌入计算失败但被静默替换为零，将引入不可见的偏差。

> **建议**：记录失败样本，在分析时排除这些样本，而非填充零向量。

**问题 4.2.3：`6_segment_analysis.py` 同样存在路径 Bug**

该脚本第 25-27 行有与 Bug 4.1.1 相同的问题，但它还从 `cleaned_results.json` 和 `matched_segment_results.json` 加载数据。如果这些文件是由同样有 Bug 的脚本生成的，结果的可信度需要整体评估。

### 4.3 可复现性问题

**问题 4.3.1：环境依赖未锁定**

`requirements.txt` 使用 `>=` 版本约束（如 `torch>=2.0`），不同时间安装可能得到不同版本，尤其是 `laion-clap` 和 `msclap` 的版本可能影响模型权重和 API。

> **建议**：提供 `requirements-lock.txt` 或使用 `pip freeze` 记录精确版本。

**问题 4.3.2：缺少端到端运行脚本**

12 个脚本需要按顺序手动运行，且某些脚本需要 GPU、某些需要 API Key。缺少一个 `run_all.sh` 或 Makefile 来管理整个流水线。

---

## 5. 实验设计层面的问题

### 5.1 模型覆盖不足

**问题 5.1.1：仅测试 3 个 CLAP 变体**

当前测试的模型：

| 模型 | 维度 | 训练数据 | 发布时间 |
|------|------|----------|----------|
| LAION CLAP (standard) | 512 | LAION-Audio-630K (英语为主) | 2023 |
| LAION CLAP (fused) | 512 | 同上 + 融合层 | 2023 |
| Microsoft CLAP | 1024 | MusicCaps + 多语言 | 2023 |

这三个模型都属于 CLAP 架构家族（HTSAT 音频编码器 + 文本编码器），且均发布于 2023 年。论文声称发现了"CLAP 模型的行为模式"，但没有测试：

- **更大的 CLAP 模型**：`laion/larger_clap_music`（音乐专用）、`laion/larger_clap_general`
- **非 CLAP 架构的音频-文本模型**：ImageBind（Meta，6 模态）、MuLan（Google，音乐-语言）、AudioLDM2 中的音频编码器
- **2024-2025 年的新模型**：音频-文本对齐领域发展迅速，2023 年的模型可能已不代表最新水平
- **纯音频特征模型**：如 MERT（音乐理解）、EnCodec（音频编解码），可作为对照

> **后续建议**：至少增加 2 个非 CLAP 架构的模型（ImageBind + 一个 2024 年模型），以支撑"跨架构"的结论。

**问题 5.1.2：模型间比较不公平**

LAION CLAP 处理完整文本，而 MS-CLAP 截断为 500 字符。虽然论文做了截断控制实验，但基础比较本身不在同一条件下进行，影响了表 2 的直接可比性。

### 5.2 语义隔离不充分

**问题 5.2.1：C0 vs C1 的混杂变量**

哈基米歌词与原始歌词的差异不仅是语义——它们还在以下维度上系统性不同：

| 维度 | 原始歌词 (C0) | 哈基米歌词 (C1) |
|------|---------------|-----------------|
| 语义 | 有意义 | 无意义 |
| 音节结构 | 多样 | 重复（ha-ji-mi） |
| 韵律规律性 | 中等 | 高 |
| 字符多样性 | 高 | 低 |
| 分词结果 | 正常 | 异常 |
| 文本长度 | 长（均值 929 字符） | 较短 |

论文承认了这一问题（"confounds semantics with phonology"），但声称 C0 vs C8 提供了更干净的语义测试。然而 C0 ≈ C8 的结果只能说明"在保持相似表面形式的前提下更换词语不影响对齐"，这与"语义不重要"是不同的论断。

**问题 5.2.2：缺少 TTS 控制实验**

隔离语义与语音的金标准方法是：使用 TTS 系统从**相同文本**生成标准化语音，然后比较不同文本条件下的对齐度。这样声学信号被标准化，差异只来自文本。

当前所有实验都使用自然录制的音频，因此无法排除音频侧的差异（如不同歌手的发音习惯、混音风格等）对结果的影响。

### 5.3 检索实验设计不足

**问题 5.3.1：当前检索是条件排名，非真正检索**

`retrieval_experiment.py` 做的是：对每首歌，将 9 个文本条件按对齐度排名，看哪个条件排第一。这回答的问题是"哪种文本与这首歌的音频最匹配？"

真正的检索实验应该是：给定一段文本（如某首歌的原始歌词），从 166 首歌的音频库中检索出正确的歌曲。这回答的问题是"CLAP 能否通过文本找到对应的音频？"

后者是一个更强的测试，也更贴近实际应用场景（音乐检索系统）。

> **后续建议**：设计 text→audio 和 audio→text 的双向检索实验，报告 Recall@1/5/10 和 MRR。

### 5.4 人工评估缺失

**问题 5.4.1：释义质量仅有自动化指标**

论文使用 sentence-transformer 相似度（0.84）和字符 Jaccard 重叠（0.37）来验证释义质量，但缺少人工评估。自动化指标无法衡量：

- 释义的"可唱性"（singability）是否与原文相当？
- 释义是否保留了原文的情感色调？
- 释义在语法和自然度上是否合格？

论文仓库中已有完整的标注工具（`scripts/annotation/annotate.py`，Flask 界面，8 个评分维度 + 成对比较），但 `REVIEW_STATE.json` 显示人工评估仍为 `pending`。

> **后续建议**：使用现有标注工具完成至少 50 首歌、2 名标注者的评估，报告 Cohen's Kappa。

---

## 6. 后续实验设计方案

以下为按优先级排列的具体实验方案，供后续研究者执行。

### 实验 A：代码修复与结果验证（优先级 P0，阻塞性）

**目标**：确保当前所有已报告结果的可靠性。

**步骤**：

1. **修复路径构造 Bug**：将所有 6 个受影响脚本中的 `os.path.dirname()` 替换为 `Path(__file__).resolve().parent.parent`。

2. **修复 Fused CLAP 初始化**：在 `fused_clap_experiment.py` 中添加 `load_ckpt()` 和 `.cuda().eval()`。

3. **修复释义截断长度**：将 `1_generate_paraphrases.py:79` 的截断从 200 改为 500（与论文一致），或在论文中修改为 200。

4. **重新运行并比较**：修复后重新运行所有实验，逐个比较结果文件与现有结果是否一致。如果 Fused CLAP 结果发生变化，论文中相关结论需要更新。

5. **提取公共工具**：创建 `scripts/utils.py`，包含所有共享函数，各脚本改为导入。

**预计耗时**：2-4 小时（含重新运行实验）
**所需资源**：GPU（用于重新计算嵌入）

---

### 实验 B：扩展模型对比（优先级 P1，高影响）

**目标**：测试更多模型，支撑"跨模型"结论。

**新增模型**：

| 模型 | 来源 | 嵌入维度 | 为什么选它 |
|------|------|----------|-----------|
| LAION CLAP-Large (music) | `laion/larger_clap_music` | 512 | 更大参数、音乐专用 |
| LAION CLAP-Large (general) | `laion/larger_clap_general` | 512 | 通用音频对比 |
| ImageBind | `facebookresearch/ImageBind` | 1024 | 非 CLAP 架构，6 模态 |
| MERT-v1-95M | `m-a-p/MERT-v1-95M` | 768 | 音乐理解，无文本分支 |

**实验设计**：

```
对每个模型 M:
  1. 加载模型
  2. 计算 166 首歌的 C0/C1/C8 文本嵌入和哈基米音频嵌入
  3. 计算余弦相似度
  4. 报告:
     - C0 vs C1: Cohen's d, p 值, 95% CI
     - C0 vs C8: Cohen's d, p 值, 95% CI
     - C0 vs C1 在该模型上的效应方向是否与 LAION CLAP 一致
  5. 对于无文本分支的模型(MERT):
     - 提取音频特征
     - 训练线性探测器区分 C0/C1 音频
     - 报告分类准确率（应接近 50%，因为音频相同）
```

**统计方法**：

- 所有模型的比较使用相同的 Bonferroni 校正（校正因子 = 模型数 × 比较数）
- 使用混合效应模型（模型为随机效应、条件为固定效应）检验条件效应的跨模型一致性
- 报告 I² 统计量量化模型间异质性

**输出文件**：`results/extended_model_comparison.json`

**预计耗时**：4-6 小时
**所需资源**：GPU（NVIDIA A100 或同等级），约 20GB 显存

---

### 实验 C：真正的音频-文本检索实验（优先级 P2）

**目标**：测试 CLAP 在真实检索场景中的行为。

**实验设计**：

**C1: Text→Audio 检索**
```
对每首歌 i (i = 1...166):
  query = 歌 i 的 C0 文本（原始歌词）
  corpus = 所有 166 首歌的哈基米音频
  用 CLAP 计算 query 与 corpus 中每条音频的相似度
  记录正确歌曲的排名
  
对 C1（哈基米文本）和 C8（释义文本）重复上述过程

报告:
  - Recall@1, @5, @10
  - MRR (Mean Reciprocal Rank)
  - 三个条件的检索性能比较
```

**C2: Audio→Text 检索**
```
对每首歌 i:
  query = 歌 i 的哈基米音频
  corpus = 所有 166 首歌的 C0 文本
  记录正确文本的排名

同时以 C1 文本和 C8 文本作为 corpus 重复
```

**C3: 交叉检索**
```
query = 歌 i 的原始音频
corpus = 所有 166 首歌的哈基米音频
问：CLAP 能否通过声学相似性找到对应的哈基米版本？
```

**关键指标**：

| 实验 | 期望结果 | 如果不符合 |
|------|----------|-----------|
| C0 text→audio Recall@1 | 低（<10%）| CLAP 不能通过歌词找到对应歌曲 |
| C1 text→audio Recall@1 | 略高于 C0 | 与条件排名结果一致 |
| 原始 audio→哈基米 audio | 高（>50%）| 声学相似性被保留 |

**输出文件**：`results/true_retrieval_results.json`

**预计耗时**：2-3 小时
**所需资源**：GPU

---

### 实验 D：TTS 语音控制实验（优先级 P2）

**目标**：通过 TTS 标准化音频信号，隔离文本侧效应。

**原理**：当前实验中，C0 和 C1 的音频来自不同的人声录制。即使旋律相同，不同录制的声学差异可能影响对齐度。使用 TTS 可以消除这一混杂。

**实验设计**：

```
1. 选择一个中文 TTS 系统（推荐 edge-tts 或 PaddleSpeech）

2. 对每首歌的以下文本生成 TTS 音频:
   - C0: 原始歌词 → tts_c0.wav
   - C1: 哈基米歌词 → tts_c1.wav
   - C8: 释义歌词 → tts_c8.wav

3. 计算 CLAP 对齐度:
   - (C0 文本, tts_c0 音频): 文本与自身语音的对齐
   - (C0 文本, tts_c1 音频): 原始文本与哈基米语音的对齐
   - (C1 文本, tts_c0 音频): 哈基米文本与原始语音的对齐
   - 以此类推的 3×3 交叉矩阵

4. 与自然录制音频的结果对比
```

**分析**：

- 如果 TTS 音频下 C0 ≈ C8 仍然成立 → 效应来自文本侧
- 如果 TTS 音频下 C0 ≈ C8 不再成立 → 效应来自音频侧的差异
- TTS 音频下 C1 > C0 是否仍然成立 → 哈基米优势是否独立于音频录制

**注意事项**：

- TTS 音频是语音朗读而非歌唱，与真实歌曲有本质区别。结果的推广性有限。
- 可以考虑使用歌唱合成（Singing Voice Synthesis）系统，如 DiffSinger，效果更贴近真实场景，但配置更复杂。

**输出文件**：`results/tts_control_results.json`

**预计耗时**：3-4 小时
**所需资源**：CPU 即可（TTS 不需要 GPU），CLAP 计算需要 GPU

---

### 实验 E：人工评估（优先级 P3）

**目标**：验证 LLM 生成的释义质量，为 C0 ≈ C8 的结论提供人工佐证。

**实验设计**：

使用已有的标注工具 `scripts/annotation/annotate.py`。

**标注方案**：

- **样本**：随机抽取 50 首歌（约 30%）
- **标注者**：至少 2 名母语为中文的标注者
- **标注维度**（已在工具中实现）：
  1. 语义保留度（1-5 分）：释义是否保留了原文的含义？
  2. 自然度（1-5 分）：释义读起来自然吗？
  3. 可唱性（1-5 分）：释义能否配合原曲旋律演唱？
  4. 情感一致性（1-5 分）：释义是否保留了原文的情感？
  5. 词汇多样性（1-5 分）：释义的用词是否与原文充分不同？
  6. 语法正确性（1-5 分）
  7. 风格匹配（1-5 分）
  8. 整体质量（1-5 分）
- **成对比较**：对于每首歌，标注者在 4 个文本变体（原文/释义/同音字/哈基米）中选择最匹配音频的版本

**统计报告**：

- 各维度的均值和标准差
- 标注者间一致性（Cohen's Kappa 或 Krippendorff's alpha）
- 语义保留度与 CLAP 对齐度的相关分析

**输出文件**：`results/human_evaluation_results.json`

**预计耗时**：每位标注者约 2 小时，加上分析 1 小时
**所需资源**：无 GPU，需要标注者

---

### 实验 F：统计方法强化（优先级 P3）

**目标**：提升统计分析的严谨性。

**具体措施**：

**F1: 等效性检验（TOST）**
```python
from statsmodels.stats.weightstats import ttost_paired

# 对 C0 vs C8，检验两者是否等价（δ = 0.1 的余弦相似度范围内）
delta = 0.01  # 实际意义的最小差异阈值
p_lower, p_upper, _ = ttost_paired(c0_align, c8_align, -delta, delta)
p_tost = max(p_lower, p_upper)
# 如果 p_tost < 0.05，则可以正面声称 C0 ≈ C8
```

**F2: 贝叶斯因子（BF₁₀）**
```python
import pingouin as pg

# 对 C0 vs C1，计算支持 H₁（有差异）的贝叶斯因子
bf = pg.bayesfactor_ttest(t_stat, n, paired=True)
# BF₁₀ > 10: 强证据支持有差异
# BF₁₀ < 0.1: 强证据支持无差异
```

**F3: Cohen's d 的置信区间**
```python
from scipy.stats import nct

# 使用非中心 t 分布计算 d 的 CI
def cohens_d_ci(d, n, alpha=0.05):
    df = n - 1
    t_obs = d * np.sqrt(n)
    ncp_lo = nct.ppf(alpha/2, df, t_obs)
    ncp_hi = nct.ppf(1 - alpha/2, df, t_obs)
    return ncp_lo / np.sqrt(n), ncp_hi / np.sqrt(n)
```

**F4: 排列检验替代参数检验**
```python
def permutation_test(a, b, n_perm=10000):
    observed = np.mean(a - b)
    combined = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        perm_diff = np.mean(perm[:n] - perm[n:])
        if abs(perm_diff) >= abs(observed):
            count += 1
    return count / n_perm
```

**F5: 跨模型混合效应分析**
```python
import statsmodels.formula.api as smf

# 模型: alignment ~ condition + (1|model) + (1|song)
# 使用线性混合效应模型检验条件效应的跨模型一致性
model = smf.mixedlm(
    "alignment ~ C(condition)",
    data=df,
    groups=df["song"],
    re_formula="~C(model)"
)
```

**输出文件**：`results/enhanced_statistics.json`

**预计耗时**：2-3 小时
**所需资源**：CPU

---

### 实验 G：跨语言验证（优先级 P4，长期目标）

**目标**：验证发现是否局限于中文。

**数据收集方案**：

| 语言 | 现象 | 来源 | 预期规模 |
|------|------|------|----------|
| 日语 | 空耳（Soramimi）歌词 | Niconico / YouTube | 50-100 首 |
| 英语 | Misheard Lyrics / Mondegreen | YouTube "misheard lyrics" | 50-100 首 |
| 韩语 | 공몬（Gongmon）翻唱 | YouTube | 30-50 首 |

**关键对比**：

- 中文哈基米 vs 日语空耳 vs 英语 Misheard：效应方向是否一致？
- LAION CLAP（英语主导训练）在英语 Misheard 上是否表现不同？
- MS-CLAP（多语言训练）在各语言上的表现是否更一致？

**预计耗时**：数据收集 2-4 周，实验 1-2 天
**所需资源**：GPU，标注者

---

## 7. 论文改写方案

### 7.1 标题和框架调整

**当前标题**：_When Meaning Dissolves: What Audio-Text Alignment Models Actually Encode_

**问题**：标题暗示因果解读（"meaning dissolves" → 意义消失后模型的行为），且 "Actually Encode" 过于笃定。

**建议选项**：

- 选项 A（推荐）：_Do CLAP Models Need Meaning? Probing Audio-Text Alignment with Chinese Nonsense Lyrics_
- 选项 B：_Surface Over Semantics: How Three CLAP Variants Respond to Chinese Lyric Perturbations_
- 选项 C：_Hachimi as a Probe: Model-Dependent Sensitivity to Lyric Surface Form in Audio-Text Alignment_

框架应从"我们发现语义不重要"转变为"我们观察到 CLAP 模型对歌词表面形式的非平凡敏感性，且这种敏感性因模型而异"。

### 7.2 结构重组

**当前结构**（主文 ~9 页）：

```
1. Introduction (1.5 页)
2. Related Work (0.7 页)
3. Method (1.5 页)
4. Experiments (4 页，7 个子节全部在主文)
5. Discussion (0.8 页)
6. Conclusion (0.3 页)
```

**建议结构**：

```
1. Introduction (1.2 页)
   - 压缩，删除冗余的贡献列表
   - 贡献改为 3 点（数据集 + 核心发现 + 声学分析）

2. Related Work (0.8 页)
   - 增加音乐检索中文本查询有效性的讨论
   - 增加中文 NLP 中分词效应的相关工作

3. Dataset and Method (1.5 页)
   - 3.1 哈基米数据集
   - 3.2 退化谱（Degeneration Spectrum）
   - 3.3 模型和评估指标

4. Core Results (2.5 页) ← 聚焦核心
   - 4.1 语义内容与对齐（C0 ≈ C8 < C1，释义不变性）
   - 4.2 跨模型比较（3 个模型 + 新增模型）
   - 4.3 检索实验（条件排名 + 真正检索）

5. Control Experiments (1.5 页) ← 回应审稿人质疑
   - 5.1 同音字语音控制
   - 5.2 MS-CLAP 截断控制
   - 5.3 时间匹配控制
   - 5.4 TTS 控制（如果完成）

6. Analysis (1.0 页)
   - 6.1 声学特征预测
   - 6.2 嵌入几何（ZCA whitening）
   - 6.3 声学扰动分析

7. Discussion (1.0 页) ← 加强
   - 明确因果限制
   - 实践意义
   - 与现有文献的对话

8. Conclusion (0.3 页)

Appendix:
   - A: 条件详情
   - B: 释义质量验证（自动 + 人工）
   - C: Vocal-only 分析
   - D: 完整统计表
   - E: 效应量 CI 和 Bayes 因子
```

### 7.3 需要修正的事实性错误

| 位置 | 当前描述 | 实际情况 | 修正方案 |
|------|----------|----------|----------|
| Section 3.3 | "Qwen2.5-7B via local inference" | 代码使用 MiniMax API (MiniMax-M2.7) | 改为实际使用的模型 |
| Section 3.3 | "Lyrics exceeding 500 characters are truncated" | 代码截断为 200 字符 | 统一为实际值或重新生成 |
| Table 1 注 | N = 165 | 主文其他地方用 N = 166 | 统一（165 是有 C8 的歌曲数） |
| Section 3.4 | 描述了 HuBERT+BERT CCA 基线 | 结果部分未报告该基线 | 报告或删除 |

### 7.4 需要增加的内容

1. **等效性检验（TOST）**：为 C0 ≈ C8 提供正面统计支持
2. **效应量 CI**：为所有 Cohen's d 添加 95% CI
3. **人工评估表**：释义质量的人工评分（完成实验 E 后）
4. **扩展模型比较表**：增加 ImageBind 等新模型（完成实验 B 后）
5. **真正检索结果表**：text→audio 检索的 Recall@K 和 MRR（完成实验 C 后）
6. **统计效能分析**：报告各核心比较的事后效能
7. **Limitation 部分扩展**：更详细地讨论混杂变量和推广性限制

### 7.5 需要删除或精简的内容

1. **图表去重**：Table 1 和 Figure 1 选其一；Table 2 和 Figure 3 选其一
2. **Case Studies 段落**：两个案例研究占了较多篇幅但信息密度低，可缩减为 1-2 句
3. **Negation 段落**：结果为负面（无效应），可移到附录
4. **Appendix 中的 Permutation Test 表**：仅 2 行，可合并到主文
5. **Overview Figure**（fig_overview.png）：信息与正文重复较多，考虑精简或删除

---

## 8. 优先级与执行路线图

### 阶段总览

```
阶段 0: 代码修复 ─────────────────────── [必须，阻塞后续]
阶段 1: 统计强化 + 人工评估 ─────────── [高性价比，无需 GPU]
阶段 2: 扩展模型 + 检索实验 ─────────── [高影响，需要 GPU]
阶段 3: TTS 控制实验 ───────────────── [中等影响，需要 GPU]
阶段 4: 论文改写 ───────────────────── [整合所有结果]
阶段 5: 跨语言扩展 ─────────────────── [长期，下一篇论文]
```

### 详细时间线

| 阶段 | 任务 | 对应实验 | 预计耗时 | 资源 | 对论文评分的影响 |
|------|------|----------|----------|------|-----------------|
| **0** | 修复代码 Bug | 实验 A | 2-4h | GPU | 阻塞性（不修复则结果不可信） |
| **1a** | 统计强化 | 实验 F | 2-3h | CPU | +0.3 分 |
| **1b** | 人工评估 | 实验 E | 4-6h（含标注时间） | 无 | +0.5 分 |
| **2a** | 扩展模型对比 | 实验 B | 4-6h | GPU | +1.0 分 |
| **2b** | 真正检索实验 | 实验 C | 2-3h | GPU | +0.5 分 |
| **3** | TTS 控制 | 实验 D | 3-4h | GPU | +0.5 分 |
| **4** | 论文改写 | — | 8-16h | 无 | +0.5 分（框架改进） |
| **5** | 跨语言扩展 | 实验 G | 2-4 周 | GPU | 独立论文级别 |

### 乐观估计

如果完成阶段 0-3 + 论文改写，预期评分：

- **Workshop**: 8.5 → 9.0/10（稳固 accept）
- **Findings/Short Paper**: 7.0 → 8.0-8.5/10（solid accept）
- **Main Track**: 7.0 → 7.5-8.0/10（borderline → weak accept）

如果同时完成跨语言扩展（阶段 5），可以作为全新的跟进论文投稿顶会主会场。

---

## 附录：快速参考

### A. 代码修复清单

```bash
# 需要修复路径 Bug 的文件:
scripts/2_compute_alignment.py    # 行 24-26
scripts/3_match_segments.py       # 行 19-21
scripts/4_export_segments.py      # 行 19-21
scripts/5_matched_alignment.py    # 行 23-25
scripts/6_segment_analysis.py     # 行 25-27
scripts/7_generate_figures.py     # 行 31-33

# 需要修复初始化的文件:
scripts/fused_clap_experiment.py  # 行 53, 添加 load_ckpt() 和 .cuda().eval()

# 需要修复截断长度的文件:
scripts/1_generate_paraphrases.py # 行 79, 200 → 500 或论文改为 200

# 统一替换模式 (每个受影响文件):
# 旧: PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 新: PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

### B. 新增脚本清单

| 脚本名 | 对应实验 | 功能 |
|--------|----------|------|
| `scripts/utils.py` | A | 共享工具函数 |
| `scripts/extended_model_comparison.py` | B | 多模型对比 |
| `scripts/true_retrieval_experiment.py` | C | 真正的检索实验 |
| `scripts/tts_control_experiment.py` | D | TTS 语音控制 |
| `scripts/enhanced_statistics.py` | F | 统计方法强化 |

### C. 论文修正核对表

- [ ] 标题改为更审慎的表述
- [ ] Abstract 明确标注观察性研究
- [ ] 修正释义模型描述（Qwen → MiniMax）
- [ ] 修正截断长度描述（500 → 200 或统一）
- [ ] 统一 N 值（165 vs 166）
- [ ] 删除或报告 HuBERT+BERT 基线
- [ ] 添加 TOST 等效性检验
- [ ] 添加 Cohen's d 的 95% CI
- [ ] 添加统计效能分析
- [ ] 添加人工评估结果
- [ ] 添加扩展模型比较
- [ ] 添加真正检索结果
- [ ] 图表去重
- [ ] 精简 Case Studies
- [ ] 扩展 Discussion
- [ ] 完善 Limitations

---

> **文档维护**：本文档应随实验进展持续更新。每完成一个实验阶段，在对应章节标注完成状态和结果摘要。
