# 第一节：从 Unconditional 到 Guided Generation

## 1.1 问题转变：从 $p_{\text{data}}(x)$ 到 $p_{\text{data}}(x \mid y)$

此前我们训练的是 **unconditional generative model**：

目标：

$$
X_1 \sim p_{\text{data}}(x)
$$

训练方式（以 Conditional Flow Matching 为例）：

$$
\mathcal{L}_{\text{CFM}}(\theta)
= \mathbb{E}\left\| u_t^\theta(x) - u_t^{\text{target}}(x|z) \right\|^2
$$

采样时模拟 ODE：

$$
dX_t = u_t^\theta(X_t) dt
$$

但现在问题变了：

我们不再想“生成一张图”，而是想生成：

> “a cat baking a cake”

于是目标变为：

$$
X_1 \sim p_{\text{data}}(x \mid y)
$$

这里 $y$ 是 guiding variable，可以是：

- class label（MNIST）
- text prompt（CLIP embedding）
- 甚至 video instruction

这就是 guided generation。

---

## 1.2 Guided Flow Model 的形式

根据讲义第 5.1 节定义，一个 guided flow/diffusion model 由：

- 神经网络：

$$
u_t^\theta : (x, y, t) \mapsto u_t^\theta(x|y)
$$

- 扩散系数：

$$
\sigma_t
$$

采样过程为：

$$
dX_t = u_t^\theta(X_t|y) dt + \sigma_t dW_t
$$

若 $\sigma_t = 0$，则为 guided flow model。

关键思想：

> 把 y 作为额外输入加入 vector field。

---

## 1.3 Guided Conditional Flow Matching Objective

如果固定某个 y，则问题退化为：

$$
z \sim p_{\text{data}}(\cdot | y)
$$

于是可以直接写出 guided CFM：

$$
\mathcal{L}^{\text{guided}}_{\text{CFM}}(\theta)
=
\mathbb{E}_{(z,y)\sim p_{\text{data}}, t, x\sim p_t(\cdot|z)}
\left\|
u_t^\theta(x|y)
-
u_t^{\text{target}}(x|z)
\right\|^2
$$

核心差别：

- 原来采样 $z \sim p_{\text{data}}(z)$
- 现在采样 $(z,y) \sim p_{\text{data}}(z,y)$

这意味着：

数据加载器必须同时提供图像与标签。

此时生成过程是“理论正确”的，即：

$$
X_1 \sim p_{\text{data}}(x|y)
$$

但——

经验发现生成结果“不够贴 prompt”。

于是引出下一节。

---

# 第二节：Classifier-Free Guidance（CFG）

这一部分是整个 lecture 的核心。

---

## 2.1 关键分解：guided vector field 的结构

对于 Gaussian probability path：

$$
p_t(x|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)
$$

可以写出（讲义 eq. 65）：

$$
u_t^{\text{target}}(x|y)
=
a_t x + b_t \nabla \log p_t(x|y)
$$

利用 Bayes：

$$
\nabla \log p_t(x|y)
=
\nabla \log p_t(x)
+
\nabla \log p_t(y|x)
$$

代入可得：

$$
u_t^{\text{target}}(x|y)
=
u_t^{\text{target}}(x)
+
b_t \nabla \log p_t(y|x)
$$

结构非常关键：

guided = unguided + classifier term

其中：

$$
\nabla \log p_t(y|x)
$$

就像一个“noisy classifier”。

---

## 2.2 人类的工程直觉：放大条件项

经验发现：

模型太“听自己”，不够“听 prompt”。

于是引入 guidance scale $w > 1$：

$$
\tilde u_t(x|y)
=
u_t^{\text{target}}(x)
+
w b_t \nabla \log p_t(y|x)
$$

推导可化为：

$$
\tilde u_t(x|y)
=
(1-w) u_t^{\text{target}}(x)
+
w u_t^{\text{target}}(x|y)
$$

这就是 classifier-free guidance 的核心公式。

注意：

- $w=1$ → 原始目标
- $w>1$ → 偏离真实分布
- 但 empirically 更好

这是一个工程 heuristic，而非概率严格一致。

---

## 2.3 为什么叫 “classifier-free”

早期方法会：

- 训练一个独立 classifier
- 用其梯度做 guidance

CFG 的突破在于：

> 用同一个网络同时学有条件与无条件。

做法：

- 扩充标签集合，加一个 ∅
- 训练时以概率 $\eta$ 把 y 置为 ∅

训练目标：

$$
\mathcal{L}_{\text{CFG-CFM}}
=
\mathbb{E}
\left\|
u_t^\theta(x|y)
-
u_t^{\text{target}}(x|z)
\right\|^2
$$

其中：

$$
y \leftarrow \emptyset \quad \text{with prob. } \eta
$$

这样：

- 网络自动学会 $u_t(x|\emptyset)$
- 同时学会 $u_t(x|y)$

推理时：

$$
\tilde u_t^\theta(x|y)
=
(1-w) u_t^\theta(x|\emptyset)
+
w u_t^\theta(x|y)
$$

这是一种 elegant engineering trick。

---

## 2.4 Diffusion 情况下的 CFG

对于 score-based diffusion：

score 分解：

$$
\nabla \log p_t(x|y)
=
\nabla \log p_t(x)
+
\nabla \log p_t(y|x)
$$

同样定义：

$$
\tilde s_t(x|y)
=
(1-w) s_t(x|\emptyset)
+
w s_t(x|y)
$$

最终 SDE 为：

$$
dX_t
=
\left[
\tilde u_t^\theta(X_t|y)
+
\frac{\sigma_t^2}{2}
\tilde s_t^\theta(X_t|y)
\right] dt
+
\sigma_t dW_t
$$

结构与 flow 情况完全一致。

CFG 是跨框架的。

---

# 第三节：Neural Network Architecture

现在问题变为：

如何表示：

$$
u_t^\theta(x|y)
$$

它必须：

- 输入：$x \in \mathbb{R}^{C\times H\times W}$
- 输入：$y$
- 输入：$t$
- 输出：同维度图像

MLP 不够用。

---

## 3.1 U-Net 结构

U-Net 是 convolution-based 架构。

结构路径（讲义第 5.2）：

输入：

$$
x_t^{\text{input}} \in \mathbb{R}^{3\times 256\times 256}
$$

编码：

$$
x_t^{\text{latent}} = E(x_t)
\in
\mathbb{R}^{512\times 32\times 32}
$$

midcoder：
保持 latent 维度

解码：

$$
x_t^{\text{output}}
\in
\mathbb{R}^{3\times 256\times 256}
$$

关键机制：

- downsample 增通道
- upsample 减通道
- residual connections
- skip connections

U 形结构保证：

- global abstraction
- local detail retention

---

## 3.2 Diffusion Transformer (DiT)

DiT 放弃 convolution，纯 attention。

核心步骤：

1. 把图像切成 patches
2. 每个 patch embedding
3. self-attention

Stable Diffusion 3 使用 modified DiT。

优势：

- 更 scalable
- 更适合 text-conditioning
- 易与 cross-attention 融合

---

## 3.3 Latent Diffusion

高分辨率图像：

$$
1000 \times 1000
\Rightarrow 10^6 \text{ 维}
$$

太耗内存。

解决：

使用 autoencoder。

流程：

1. encode：

$$
x \rightarrow z
$$

2. 在 latent space 上训练 diffusion
3. decode：

$$
z \rightarrow x
$$

优点：

- memory efficient
- generative model 聚焦 semantic 结构
- 低层细节交给 decoder

几乎所有 SOTA 模型都使用 latent diffusion。

---

# 第四节：如何编码 y

这是 conditional generation 成败关键。

分两步：

---

## 4.1 Embedding Raw Input

### 情况 1：class label

学习 embedding lookup table。

简单直接。

---

### 情况 2：text prompt

使用 frozen pretrained model：

- CLIP（全局语义）
- T5-XXL（sequence-level）
- UL2 / ByT5（字符级）

可组合多 embedding。

---

## 4.2 Feeding Embedding

常见方法：

1. 线性映射到 channel 数
2. reshape 为 $C\times1\times1$
3. broadcast add 到 feature map

或者：

使用 cross-attention：

- image patches attend text tokens

这是 Stable Diffusion 3 的核心机制。

---

# 第五节：Case Study

---

## 5.1 Stable Diffusion 3

特点：

- 使用 conditional flow matching
- latent diffusion
- CFG training
- MM-DiT backbone
- 多文本 embedding（CLIP + T5）

模型规模：

8B parameters

采样：

50 steps  
CFG weight 2.0–5.0

---

## 5.2 Meta Movie Gen Video

视频空间：

$$
x \in \mathbb{R}^{T\times C\times H\times W}
$$

关键挑战：

时间维度爆炸。

解决：

Temporal Autoencoder：

$$
T', H', W'
\rightarrow
\frac{1}{8}
$$

模型 backbone：

DiT + 时空 patchification + cross-attention

文本 embedding：

- UL2
- ByT5
- MetaCLIP

规模：

30B parameters

---

# 总结结构图（逻辑主线）

1. 从 unconditional 到 conditional
2. 发现条件不够强
3. 推导 score 分解
4. 引入 guidance scale
5. 通过 label dropping 实现 single-model CFG
6. 在 flow 与 diffusion 下统一
7. 解决高维图像问题 → U-Net / DiT
8. 解决内存问题 → latent diffusion
9. 解决文本理解问题 → multi-embedding + cross-attention
10. 扩展到视频

---

这节课的真正思想核心只有一句话：

> 把条件项当成一个可放大的“分类器梯度”，  
> 并用一个模型同时学有条件和无条件，然后在推理时人为调节权重。

这是一个工程与概率论交汇的经典例子。

如果你愿意，我们可以下一步：

- 从 Fokker–Planck 角度解释为什么放大 score 会改变分布  
- 或从 optimal transport 视角解释 flow matching + CFG  
- 或讨论 guidance scale 的理论极限行为  

真正有趣的部分，现在才开始。