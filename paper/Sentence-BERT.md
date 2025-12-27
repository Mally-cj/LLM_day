📑 论文研读笔记：Sentence-BERT (SBERT)

2025-12-24 21:11:05 Wednesday 


## 1. 基础信息 (Basic Information)

- **标题 (Title)**：Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- **作者/机构 (Authors/Affiliation)**：Nils Reimers, Iryna Gurevych / 达姆施塔特工业大学 (UKP Lab)
- **年份/会议 (Year/Venue)**：2019 / EMNLP
- **关键词 (Keywords)**：Sentence Embeddings, Siamese Network, BERT, Semantic Similarity
- **文章影响力**：Semantic Scholar/Google Scholar 口径下引用量已达两万级（arXiv 页面聚合信息显示 cited by 21k+）
- **技术定位**：BERT (Cross-Encoder) $\rightarrow$ **SBERT (Bi-Encoder)** $\rightarrow$ BGE / OpenAI Embeddings
  
    > SBERT 开启了预训练模型生成高质量、高效率句子向量的时代。
    > 

---

## 2. 核心挑战与动机 (Problem & Motivation)

### 2.1 待解决的问题

- **计算效率瓶颈**：原生 BERT 在计算大规模句子对相似度时速度极慢。

### 2.2 现有方法的局限

- **Cross-Encoder 结构限制**：必须将两个句子同时输入模型，导致在 $N$ 个句子中找相似对需要 $O(n^2)$ 的复杂度（1万句需约65小时）。
- **特征表示差**：直接提取 BERT 的 `[CLS]` 或词向量平均值，在未经微调时语义聚类效果甚至不如 GloVe。

---

## 3. 技术方案 (Methodology)

### 3.1 架构改进 (Architecture)

- **输入表示**：独立的句子 A 和 句子 B 分别进入共享参数的 BERT。
- **关键结构组件**：
    - **Siamese Network (双塔)**：两个 BERT 权重共享。
    - **Pooling Layer (池化层)**：在 BERT 输出层后增加 MEAN 池化（尝试 CLS / MEAN / MAX），将 Token 向量聚合为定长句子向量。

### 3.2 训练策略与目标函数 (Objectives)

SBERT 的核心在于：通过不同的 **目标函数（Objective Functions）**，在 Siamese 网络结构上对预训练的 BERT 进行微调。

#### 1. 分类任务目标 (Classification Objective)

主要用于 **NLI（自然语言推理）** 等标签化数据集。

- **输入 (Input)**：一对句子 $(s_a, s_b)$及其类别标签 $y$（如：蕴含、矛盾、中立）。
- **计算流程 (Forward Pass)**：
    1. 分别得到句子向量 $u$ 和 $v$。
    2. 构造特征向量：将 $u$、$v$ 和它们的元素级差异 $\|u - v\|$ 进行拼接。
- **输出 (Output)**：一个经过映射后的概率分布 $\hat{y}$：
  
    $$
    \hat{y} = \text{softmax}\big(W_t(u, v, \lvert u - v \rvert)\big)
    $$
    
    其中 $W_t \in \mathbb{R}^{3n \times k}$，$n$ 是向量维度，$k$ 是类别数。
    
- **损失函数 (Loss Function)**：交叉熵损失 (Cross-Entropy Loss)：
  
    $$
    \mathcal{L} = -\sum y \log(\hat{y})
    $$
    

---

#### 2. 回归任务目标 (Regression Objective)

主要用于 **STS（语义文本相似度）** 等连续评分数据集。

- **输入 (Input)**：一对句子 $(s_a, s_b)$ 及其相似度分数 $y$（通常归一化到 $[0, 1]$）。
- **计算流程 (Forward Pass)**：
    1. 分别得到句子向量 $u$ 和 $v$。
    2. 直接计算两个向量的余弦相似度。
- **输出 (Output)**：一个相似度预测值：
  
    $$
    \text{score} = \cos(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}
    $$
    
- **损失函数 (Loss Function)**：均方误差损失 (MSE Loss)：
  
    $$
    \mathcal{L} = \big\lVert y - \cos(u, v) \big\rVert^2
    $$
    

---

#### 3. 三元组任务目标 (Triplet Objective)

主要用于大规模检索和度量学习。

- **输入 (Input)**：三个句子构成的三元组 $(a, p, n)$。
    - $a$ (Anchor)：锚点句子。
    - $p$ (Positive)：与锚点相似的句子。
    - $n$ (Negative)：与锚点不相似的句子。
- **计算流程 (Forward Pass)**：
    1. 分别计算三个句子的向量 $s_a, s_p, s_n$。
    2. 计算欧氏距离：$d_+ = \|s_a - s_p\|$，$d_- = \|s_a - s_n\|$。
- **输出 (Output)**：模型并不输出单一标量，而是通过约束向量间的距离关系来学习表征。
- **损失函数 (Loss Function)**：三元组损失 (Triplet Loss)：
  
    $$
    \mathcal{L} = \max\big(0,\ \epsilon + \|s_a - s_p\| - \|s_a - s_n\|\big)
    $$
    
    其中 $\epsilon$ 是边距（Margin），确保正例比反例至少近 $\epsilon$ 的距离。
    



---





## 4. 实验验证 (Experiments)

> **这篇论文一共做了 5 类核心实验**，分别从
> **“句向量是否可用” → “不同训练信号的作用” → “能不能迁移” → “能不能排序” → “效率是否真的提升”**
> 五个维度，锁死 SBERT 的工程合理性。

---

### 实验 1：BERT 原生句向量到底行不行？（对照实验）

#### 实验目的

回答一个最基础、但必须先回答的问题：

> **“直接用 BERT 的 CLS / mean pooling，当 sentence embedding 好不好用？”**

这是 SBERT 一切动机的起点。

#### 实验做法

* 不训练（或只用原始预训练）
* 对比多种句向量方式：

  * BERT-CLS
  * BERT-mean
  * BERT-max
* 在 **STS 任务**上用 cosine 相似度评估

#### 结论

* 原生 BERT 句向量 **效果很差**
* 甚至不如一些传统 sentence embedding 方法

👉 **直接否定“BERT 自带句向量可用”的假设**

这是 SBERT 必须“重新训练”的前提。

---

### 实验 2：SBERT 训练后，STS 上效果如何？（核心主结果）

#### 实验目的

验证：

> **“如果用双塔 + 合适的训练目标，sentence embedding 能不能真的学到语义相似度？”**

#### 实验做法

* 用双塔结构
* 在不同训练数据/目标上训练：

  * NLI
  * STS
  * NLI + STS
* 在多个 STS 数据集上评估（Spearman ρ）

#### 对比对象

* 原生 BERT 句向量
* 传统 embedding（InferSent、USE 等）
* BERT cross-encoder（作为上限参考）

#### 结论

* **SBERT 在“可扩展设置（cosine）”下显著优于原生 BERT 句向量**
* 和 cross-encoder (把句子 A 和句子 B 拼接后，一起输入 BERT，用 [CLS] 做回归/分类的标准 BERT 用法)有差距，但差距在可接受范围内

👉 证明：
**双塔不是“为了快牺牲一切”，而是“快 + 还能用”**

---

### 实验 3：NLI / STS / Triplet 训练信号各自有什么作用？（训练目标对比 + 消融）

#### 实验目的

回答你前面问的那个关键问题：

> **“为什么要做 3 个训练任务？各自到底在干什么？”**

#### 实验做法

* 固定双塔结构
* 分别用：

  * 只用 NLI
  * 只用 STS
  * NLI + STS
  * Triplet
* 对比 STS 性能、迁移性能

#### 结论

* **NLI**：给 embedding 注入通用语义结构（泛化好）
* **STS**：直接对齐相似度刻度（任务对齐最好）
* **Triplet**：排序/检索导向明显

👉 这是在说明：
**不同训练信号 = embedding 空间的不同“约束维度”**

---

### 实验 4：Sentence Embedding 能不能迁移？（SentEval）

#### 实验目的

验证：

> **“SBERT 学到的不是 STS 特化技巧，而是通用句向量”**

#### 实验做法

* 固定 embedding，不微调
* 在 SentEval 的 7 个下游分类任务上测试
* 用简单分类器（如 logistic regression）

#### 对比

* 原生 BERT 句向量
* 其他 sentence embedding 方法

#### 结论

* SBERT 在多数迁移任务上显著优于原生 BERT 句向量
* 说明 embedding **不是只对 STS 有效**

👉 这是**“泛化能力”实验**。

---

### 实验 5：Triplet 排序实验（检索一致性）

#### 实验目的

单独验证：

> **“在检索/排序语义下，embedding 排序是否正确？”**

#### 实验做法

* 使用 Wikipedia section triplets
* 评估：

  * anchor–positive 是否比 anchor–negative 更近

#### 结论

* 双塔 + triplet loss 在排序一致性上明显更好

👉 这是**最贴近真实搜索/召回场景的实验**。

---

### 实验 6：效率与工程可行性实验（非常关键）

#### 实验目的

锁死论文的“工程价值”，而不是学术炫技：

> **“SBERT 在真实规模下，能不能用？”**

#### 实验做法

* 对比：

  * BERT cross-encoder
  * SBERT bi-encoder
* 任务：

  * 语义搜索
  * 大规模相似度计算
* 指标：

  * 总耗时
  * 推理速度

#### 结论

* 65 小时 → 几秒
* 复杂度从 O(N × BERT) → O(BERT + N × cosine)

👉 **这是 SBERT 这篇论文“存在的根本理由”**

---

## 5. 总结与反思 (Conclusion & Reflections)

### 5.1 3个训练任务是要干什么
| 任务      | 约束的是什么 | 解决什么问题            |
| ------- | ------ | ----------------- |
| NLI     | 全局语义结构 | embedding 是否“懂语义” |
| STS     | 距离刻度   | cosine 有没有意义      |
| Triplet | 局部排序   | 检索排得准不准           |



