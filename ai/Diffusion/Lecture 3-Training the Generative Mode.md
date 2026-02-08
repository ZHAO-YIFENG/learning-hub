## 一、核心目标：如何把一个简单分布“推”成数据分布？

所有 flow / diffusion 方法的目标，都可以表述为同一句话：

> **构造一个连续时间的随机过程，使其分布从一个容易采样的初始分布 $p_{\text{init}}$ 平滑演化到数据分布 $p_{\text{data}}$**。

两份 PDF 反复强调：  
关键不在于“噪声”或“去噪”本身，而在于 **概率分布随时间的演化路径（probability path）是否被正确跟随**。

---

## 二、概率路径（Probability Path）：训练问题的几何骨架

### 1. Conditional probability path：以单个数据点为锚

- 定义：  
  $$p_t(x \mid z)$$
  表示在时间 $t$，从数据点 $z$ 出发的**条件分布路径**。
- 直觉：  
  它刻画了“**如果我最终想生成 $z$，中间状态 $x_t$ 会长什么样**”。

**Gaussian 例子（贯穿全文的标准路径）**：
$$
p_t(x \mid z) = \mathcal N(\alpha_t z,\;\beta_t^2 I)
$$

这里：
- $\alpha_t$：保留多少数据结构  
- $\beta_t$：注入多少噪声

> 这一层是“可控的、可解析的”，也是后续一切可训练性的来源。

---

### 2. Marginal probability path：真正想要“跟随”的对象

- 定义：
  $$
  p_t(x) = \int p_t(x \mid z)\, p_{\text{data}}(z)\, dz
  $$
- 含义：
  - 它描述了**整体分布**在时间 $t$ 的形态
  - 满足边界条件：
    - $p_0 = p_{\text{init}}$
    - $p_1 = p_{\text{data}}$

但问题在于：  
👉 **这个 marginal path 通常是不可解析的**。

---

## 三、Vector Field：让 ODE“跟着分布走”

### 1. Conditional vector field：可写公式，但“带条件”

- 定义：
  $$
  u_t^{\text{target}}(x \mid z)
  $$
- 关键性质：
  > 若 ODE 使用该向量场，则轨迹的分布正好跟随 $p_t(x\mid z)$

在 Gaussian 路径下，有显式表达式：
$$
u_t^{\text{target}}(x \mid z)
= \Big(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\Big) z
+ \frac{\dot\beta_t}{\beta_t} x
$$

这一点非常重要：  
**conditional 量是 tractable 的**。

---

### 2. Marginal vector field：真正想学，但算不出来

- 定义：
  $$
  u_t^{\text{target}}(x)
  = \int u_t^{\text{target}}(x \mid z)\,
  \frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)} dz
  $$

- 含义：
  - 它定义了一个 ODE：
    $$
    dX_t = u_t^{\text{target}}(X_t)\,dt
    $$
  - 该 ODE 的解分布满足：
    $$
    X_t \sim p_t
    $$

**但致命问题是：这个积分不可计算。**

---

## 四、Flow Matching：用“条件”间接学习“边缘”

### 1. 理想目标：拟合 marginal vector field

理想损失：
$$
\mathcal L_{\text{FM}}
= \mathbb E_{t,x\sim p_t}\|u_t^\theta(x)-u_t^{\text{target}}(x)\|^2
$$

👉 不可用，因为 $u_t^{\text{target}}(x)$ 不可算。

---

### 2. 关键技巧：Conditional Flow Matching

定义**可计算的替代损失**：
$$
\mathcal L_{\text{CFM}}
= \mathbb E_{t,z,x\sim p_t(\cdot|z)}
\|u_t^\theta(x)-u_t^{\text{target}}(x|z)\|^2
$$

**核心定理（Theorem 18）**：
> $$
> \mathcal L_{\text{FM}}(\theta)
> = \mathcal L_{\text{CFM}}(\theta) + C
> $$
> 常数 $C$ 与参数 $\theta$ 无关，因此两者梯度完全一致。

**概念含义**：
- 虽然我们在“对着 conditional 学”
- 但在期望意义下，**等价于在学习 marginal**

这是整套 flow matching 理论的支点。

---

### 3. 训练流程（Gaussian CondOT）

$$
x_t = \alpha_t z + \beta_t \varepsilon,\quad \varepsilon\sim\mathcal N(0,I)
$$

损失退化为极其简单的形式：
$$
\|u_t^\theta(x_t) - (\dot\alpha_t z + \dot\beta_t \varepsilon)\|^2
$$

> 这解释了为什么 **Stable Diffusion 3、MovieGen** 能用如此“简单”的 loss 训练出来。

---

## 五、Score Matching：把 ODE 扩展成 SDE

### 1. 为什么引入 score？

为了构造 **随机采样（SDE）**，需要知道分布的梯度信息：
$$
\nabla \log p_t(x)
$$

它描述了：  
> **在点 $x$ 处，概率密度增长最快的方向**

---

### 2. 同样的套路：conditional → marginal

- 定义 conditional score：
  $$
  \nabla \log p_t(x\mid z)
  $$
- 定义 conditional score matching loss：
  $$
  \mathcal L_{\text{CSM}}
  = \mathbb E\|s_t^\theta(x)-\nabla\log p_t(x|z)\|^2
  $$

**Theorem 20**：
$$
\mathcal L_{\text{SM}}(\theta)
= \mathcal L_{\text{CSM}}(\theta)+C
$$

逻辑结构与 flow matching **完全同构**。

---

### 3. Gaussian 情形 → Denoising Diffusion

Gaussian conditional score：
$$
\nabla \log p_t(x|z)
= -\frac{x-\alpha_t z}{\beta_t^2}
$$

重参数化后，训练目标变成：
$$
\|\varepsilon_\theta(x_t,t)-\varepsilon\|^2
$$

这就是 **DDPM / diffusion model** 中熟悉的“预测噪声”训练方式。

---

## 六、一个关键事实：Gaussian 路径下，Score 和 Flow 是等价的

**Conversion Formula（极其重要）**：

$$
u_t(x)
= \Big(\frac{\beta_t^2\dot\alpha_t}{\alpha_t}-\beta_t\dot\beta_t\Big)\,
\nabla\log p_t(x)
+ \frac{\dot\alpha_t}{\alpha_t}x
$$

含义：
- 在 Gaussian probability path 下：
  - 学会 **score** ⇔ 学会 **vector field**
- 因此：
  - **不需要同时训练两张网络**

这解释了：
> 早期 diffusion 只做 score matching，仍然能隐式确定 ODE / SDE。

---

## 七、统一视角总结（从“方法”回到“结构”）

- **Flow Matching**
  - 学 $u_t$
  - 采样：ODE（确定性）
- **Diffusion Model**
  - 学 score 或 noise
  - 采样：SDE / Probability Flow ODE
- **Gaussian DDM**
  - 只是 probability path 的一个特例
- **真正的核心不是噪声，而是：**
  > 如何构造 + 跟随一条概率路径
---

## 八、补充：扩散模型文献导读

这一部分的目的不是引入新技术细节，而是**统一不同扩散模型表述之间的关系**，解释为什么文献中看起来“方法很多”，但本质上高度一致。

---

### 1. 离散时间（Discrete-time）vs 连续时间（Continuous-time）

- **早期扩散模型（DDPM 等）**
  - 使用离散时间 Markov chain：$t = 0,1,2,\dots,T$
  - 训练目标通过 **ELBO（Evidence Lower Bound）** 构造
  - ELBO 本质上只是目标 loss 的一个下界

- **连续时间视角（SDE / ODE）**
  - 后续工作表明：离散扩散是连续时间 SDE 的近似
  - 在连续时间下：
    - ELBO 变为**严格等价的目标**
    - flow matching / score matching 的定理是**等式而非下界**
  - 优势：
    - 数学上更干净
    - 数值误差可通过 ODE / SDE 求解器控制

**结论**：  
离散扩散与连续扩散**不是本质不同的方法**，而是同一理论在不同时间刻度下的实现。

---

### 2. Forward process vs Probability Path

- **Forward process（早期 diffusion 文献）**
  - 从数据点 $z$ 出发，构造一个“加噪”SDE：
    $$
    \bar X_0 = z,\quad
    d\bar X_t = u^{\text{forw}}_t(\bar X_t)\,dt + \sigma^{\text{forw}}_t\,dW_t
    $$
  - 设计使得 $t\to\infty$ 时：
    $$
    \bar X_t \sim \mathcal N(0,I)
    $$

- **Probability path 视角（flow matching）**
  - 不再关心 forward SDE 是否真的被“模拟”
  - 只关心：
    $$
    p_t(x\mid z)
    $$
    是否是一个可采样、可解析的路径
  - Forward process 只是**构造 Gaussian probability path 的一种方式**

**关键限制**：
- 为了能解析 $p_t(x\mid z)$，forward process 中的向量场必须是**仿射形式**
- 这导致经典 diffusion 只能使用 **Gaussian probability path**

---

### 3. 时间反演（Time-reversal）vs Fokker–Planck 视角

- **经典扩散模型推导**
  - 通过对 forward SDE 做 **time-reversal**
  - 得到反向生成过程

- **本课程 / flow matching 的视角**
  - 直接从：
    - Fokker–Planck 方程
    - Continuity equation
  - 构造：
    - $u_t^{\text{target}}$
    - $\nabla\log p_t$

- **重要观察**
  - 对生成任务而言，通常只关心最终样本 $X_1$
  - 是否严格是 time-reversal **并不重要**
  - 实践中：
    - Probability flow ODE 往往比 time-reversal SDE 更稳定、效果更好

**结论**：  
Time-reversal 是一种历史推导路径，而不是必要工具。

---

### 4. Flow Matching 与 Stochastic Interpolants 的位置

- **Flow Matching**
  - 只使用 ODE（纯 flow）
  - 采样是确定性的
  - 关键创新：
    - 不需要 forward process
    - 仍可 scalable 训练

- **Stochastic Interpolants（SI）**
  - 同时涵盖：
    - 纯 flow（ODE）
    - 含噪情形（SDE / Langevin dynamics）
  - 从插值函数的角度构造 probability path

- **相对 diffusion models 的优势**
  - 不局限于 Gaussian 初始分布
  - 不局限于 Gaussian probability path
  - 理论上允许：
    $$
    p_{\text{init}} \;\longrightarrow\; p_{\text{data}}
    $$
    的**任意分布到任意分布**变换

---

### 5. 文献脉络的统一总结

- 文献中的差异主要来自：
  1. 时间是离散还是连续  
  2. 是否使用 forward process  
  3. 是否采用 time-reversal 作为推导工具  
  4. 使用 ODE 还是 SDE 进行采样  

- 在本笔记的统一框架下：
  - **Diffusion models**  
    = 使用 Gaussian probability path 的特殊情形
  - **Denoising diffusion models**  
    = Gaussian path + score matching
  - **Flow matching / stochastic interpolants**  
    = 更一般的 probability path 训练框架

> 不同论文的“表面差异”，在概率路径 + 向量场 / score 的视角下，都可以被看作同一结构的不同坐标系表示。

---



