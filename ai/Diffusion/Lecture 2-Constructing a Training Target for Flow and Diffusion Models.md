
## 一、问题背景：为什么需要“Training Target”？

### 1. Flow / Diffusion 的共同建模视角

无论是 **Flow model（ODE）** 还是 **Diffusion model（SDE）**，核心结构是一样的：

- 从一个**简单初始分布**  
  $$
  X_0 \sim p_{\text{init}} \quad (\text{通常是 Gaussian})
  $$
- 通过一个 **神经网络参数化的向量场** $u_t^\theta(x)$，
- 沿时间 $t \in [0,1]$ 演化，最终希望
  $$
  X_1 \sim p_{\text{data}}
  $$

Flow（ODE）：
$$
dX_t = u_t^\theta(X_t)\,dt
$$

Diffusion（SDE）：
$$
dX_t = u_t^\theta(X_t)\,dt + \sigma_t dW_t
$$

👉 **关键目标**：学到一个向量场，使“从噪声到数据”的演化在分布层面是正确的。

---

### 2. 训练的困难：没有“标签”

在普通监督学习中：

- 输入 $x$ → 标签 $y$
- loss = prediction vs label

但这里：

- 我们**没有**“正确的向量场标签”
- 只能看到：  
  “如果这个向量场是对的，那么分布应该被正确地 transport”

因此，训练目标被写成：
$$
\mathcal{L}(\theta)
= \mathbb{E}\,\bigl\|u_t^\theta(x) - u_t^{\text{target}}(x)\bigr\|^2
$$

**问题变成：**  
👉 如何**构造**这个 $u_t^{\text{target}}(x)$？

---

## 二、核心思想：从「概率路径」反推「向量场」

整门课的逻辑主线其实是：

> **先设计分布如何随时间变化（Probability Path），  
> 再反推出一个向量场，使 ODE / SDE 的解服从这个分布路径。**

---

## 三、Probability Path：从 Noise 到 Data 的“分布轨迹”

### 1. Conditional vs Marginal（非常关键的区分）

- **Conditional（条件）**：  
  “针对单个数据点 $z$”
- **Marginal（边缘）**：  
  “对所有数据点的总体分布”

这是整套推导的结构骨架。

---

### 2. Conditional Probability Path $p_t(x \mid z)$

定义：  
$$
p_0(\cdot\mid z) = p_{\text{init}}, 
\qquad
p_1(\cdot\mid z) = \delta_z
$$

含义：

- 对于**固定的数据点** $z$
- 分布从“噪声”逐渐收缩到“确定点 $z$”

> 可以把它理解为：  
> **“如果目标是生成这个 $z$，那噪声应该怎么一步步变成它？”**

---

### 3. Marginal Probability Path $p_t(x)$

由 conditional 路径诱导而来：
$$
p_t(x)
=
\int p_t(x\mid z)\,p_{\text{data}}(z)\,dz
$$

含义：

- 先采样一个真实数据 $z\sim p_{\text{data}}$
- 再沿着对应的 conditional path 采样
- 得到整体分布的时间演化

满足：
$$
p_0 = p_{\text{init}}, \quad p_1 = p_{\text{data}}
$$

---

## 四、Gaussian Probability Path（最重要的具体例子）

这是 diffusion / flow matching 中**最核心**的选择。

### 1. 定义

给定噪声调度函数 $\alpha_t, \beta_t$：
$$
p_t(x\mid z) = \mathcal{N}(\alpha_t z,\; \beta_t^2 I)
$$

约束：
$$
\alpha_0 = 0,\ \beta_0 = 1;\quad
\alpha_1 = 1,\ \beta_1 = 0
$$

直觉：

- $t=0$：纯噪声
- $t=1$：退化为 $\delta_z$
- 中间：线性均值 + 逐渐减小方差

---

### 2. 采样形式（很重要）

$$
z\sim p_{\text{data}},\ \varepsilon\sim\mathcal{N}(0,I)
\quad\Rightarrow\quad
x_t = \alpha_t z + \beta_t \varepsilon
$$

这也是 diffusion 中常见的 forward noising 结构。

---

## 五、从 Probability Path 到 Vector Field

### 1. Conditional Vector Field $u_t^{\text{target}}(x\mid z)$

目标：

> 设计一个向量场，使得 ODE 解的分布 **正好等于**
> $$
> X_t \sim p_t(\cdot\mid z)
> $$

对于 Gaussian path，可显式写出：
$$
u_t^{\text{target}}(x\mid z)
=
\Bigl(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\Bigr) z
+
\frac{\dot\beta_t}{\beta_t}x
$$

**理解要点：**

- 对 $z$：负责“朝目标点拉”
- 对 $x$：负责整体收缩 / 扩散
- 是一个**线性向量场**

---

### 2. Marginalization Trick（全课最关键定理）

如果：

- 每个 $z$ 都有一个正确的 conditional vector field
- 那么整体的 marginal vector field 是它们的**条件期望**

$$
u_t^{\text{target}}(x)
=
\int
u_t^{\text{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)}
\,dz
$$

含义：

> **训练 flow = 学这个 marginal vector field**

而不是关心每个 $z$。

---

## 六、Continuity Equation：ODE 为什么能“搬运分布”

对于 ODE：
$$
dX_t = u_t(X_t)\,dt
$$

“轨迹服从分布路径”  
⇔  
概率密度满足 PDE：
$$
\partial_t p_t(x)
=
-\nabla\cdot\bigl(p_t(x)u_t(x)\bigr)
$$

直觉解释：

- 左边：某点概率密度随时间变化
- 右边：向量场导致的“流入 − 流出”
- 本质是**概率质量守恒**

这条方程是 **Flow Matching 理论基础**。

---

## 七、扩展到 Diffusion：Score Function 登场

### 1. SDE 形式

$$
dX_t
=
\Bigl[
u_t^{\text{target}}(X_t)
+
\frac{\sigma_t^2}{2}\nabla\log p_t(X_t)
\Bigr]dt
+
\sigma_t dW_t
$$

新增项：

- $\nabla\log p_t(x)$：**Score function**
- 补偿随机扩散带来的概率扩散效应

---

### 2. Gaussian Path 的 Score

对于：
$$
p_t(x\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I)
$$

有解析解：
$$
\nabla\log p_t(x\mid z)
=
-\frac{x-\alpha_t z}{\beta_t^2}
$$

Marginal score 同样是 conditional score 的加权平均。

---

## 八、统一视角：Flow Matching vs Score Matching

- **Flow model**
  - 学：marginal vector field $u_t^{\text{target}}$
  - Loss：MSE（Flow Matching）

- **Diffusion model**
  - 学：marginal score $\nabla\log p_t$
  - Loss：Score Matching

但二者：

> **共享同一条 probability path**

---

## 九、整节课的“一条主线总结”

> **设计分布如何从 noise 演化到 data  
> → 用 PDE（continuity / Fokker–Planck）反推出向量场  
> → 把这个向量场当作训练 target  
> → 用 MSE 学它**

---

## 十、你真正需要“记住”的 6 个公式（课程原话）

1. Conditional probability path $p_t(x\mid z)$

2. Marginal probability path $p_t(x)$

3. Conditional vector field $u_t^{\text{target}}(x\mid z)$

4. Marginal vector field $u_t^{\text{target}}(x)$

5. Conditional score $\nabla\log p_t(x\mid z)$

6. Marginal score $\nabla\log p_t(x)$

其余推导，**理解思路即可，不必死记**。
