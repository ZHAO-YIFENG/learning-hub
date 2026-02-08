
# 从「生成」到「采样」，再到「ODE / SDE 生成模型」

---
## 一、生成式建模的核心重述：**Generation = Sampling**

### 1️⃣ 为什么“生成”必须被重新定义？

直觉上的“好图 / 坏图 / 像不像狗”是**主观判断**，无法直接用于建模。  
课程的第一步是：**把主观质量转化为概率问题**。

> 核心转译：  
> **“这张图有多好” ≈ “在真实数据中出现的概率有多大”**

---

### 2️⃣ 数据对象的统一表示：**Objects as Vectors**

- 图片、视频、蛋白质结构看似不同
- 在建模层面，统一抽象为一个高维向量
\[
p_{\text{data}}(z)
\]
$$
z \in \mathbb{R}^d  
$$
- Image：$z \in \mathbb{R}^{H \times W \times 3}$ 
- Video： $z \in \mathbb{R}^{T \times H \times W \times 3}$
- Molecule： $z \in \mathbb{R}^{3 \times N}$ 

📌 **重要限定**：  
本课程**只讨论连续数据**（文本这种离散序列暂不涉及）

---
### 3️⃣ Generation as Sampling（关键逻辑跳跃）

- 假设存在一个**真实但未知的分布**：  
$$
  p_{\text{data}}(z)  
$$

- “生成一个样本”   
$$
z \sim p_{\text{data}}
$$
这一步非常关键，它把生成问题**彻底转成概率采样问题**。

---
### 4️⃣ Dataset 的角色

- 我们并不知道 $p_{\text{data}}$ 的解析形式
- 只能通过 **有限样本集** 近似它：

$$

{z_1, \dots, z_N} \sim p_{\text{data}}  

$$
👉 **训练 = 用有限样本反推生成机制**

---

### 5️⃣ Conditional Generation（提前埋下的伏笔）

- 条件生成 = 从条件分布采样：
$$
z \sim p_{\text{data}}(\cdot \mid y)  
$$
-  y 可以是：
    - 文本 prompt
    - 标签
    - 其他模态信息

📌 **课程策略**：

> 先解决 _unconditional_，再推广到 _conditional_

---

## 二、生成的“机制”：从 Noise 到 Data

### 1️⃣ 为什么要引入一个“简单分布”？

直接采样 $p_{\text{data}}$ 做不到，于是引入：

$$
p_{\text{init}} \quad \text{（通常是 } \mathcal{N}(0, I_d)\text{）}  
$$
生成模型的目标被重新表述为：

> **学一个“连续变换”，把噪声分布变成数据分布**

---

## 三、Flow Models：用 **ODE** 做生成

---

### 1️⃣ ODE 的三件套：Vector Field → Trajectory → Flow

#### （1）Vector Field（速度场）

$$
u_t(x): \mathbb{R}^d \times [0,1] \to \mathbb{R}^d  
$$

- 给定位置 x 和时间 t
- 返回一个“往哪走、走多快”的向量

---

#### （2）ODE（轨迹定义）

$$
\frac{dX_t}{dt} = u_t(X_t), \quad X_0 = x_0  
$$

含义非常直白：

> **当前位置的变化率 = 当前位置对应的速度**

---

#### （3）Flow（解的整体映射）

$$
\psi_t(x_0) = X_t  
$$

- Flow 是一个 **整体空间变形**
- 在理论上是 **diffeomorphism（可逆、光滑）**

📌 **重要结论（存在唯一性）**  
只要 $u_t$ 光滑、Lipschitz（神经网络通常满足）：

> ✔ 解存在  
> ✔ 解唯一  
> ✔ Flow 可逆

---

### 2️⃣ Flow Model 的生成逻辑

1. 从噪声采样：  
$$
X_0 \sim p_{\text{init}}  
$$
2. 用神经网络参数化 vector field：  
$$
    u^\theta_t(x)  
$$

3. 数值积分 ODE（Euler / Heun）

4. 取终点：  
$$
    X_1 \sim p_{\text{data}}  
$$

📌 **关键认知点**：

> 网络学的是 **vector field**，不是 flow 本身

---

### 3️⃣ 数值模拟（Euler Method）

$$
X_{t+h} = X_t + h , u^\theta_t(X_t)  
$$

- 每一步：**沿速度场走一小步**
- 步数 ↑ ⇒ 精度 ↑，但计算 ↑

---

## 四、Diffusion Models：用 **SDE** 做生成

---

### 1️⃣ 为什么引入随机性？

ODE 是**确定性的**：

- 同一个 $X_0$ → 同一个 $X_1$

但真实数据生成本身是**随机的**，于是引入噪声 → **SDE**

---

### 2️⃣ Brownian Motion（噪声的数学化）

布朗运动 $W_t$ 的两个关键性质：

1. **正态增量**：  
$$W_t - W_s \sim \mathcal{N}(0, (t-s)I)$$
2. **独立增量**

	数值近似：
$$W_{t+h} = W_t + \sqrt{h}*\varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)$$  
---

### 3️⃣ SDE 的核心形式

$$
dX_t = u_t(X_t)*dt + \sigma_t* dW_t  
$$

- ($u_t$)：**drift（确定性趋势）**
- ($\sigma_t$)：**diffusion coefficient（噪声强度）**

📌 记住一句话就够了：

> **SDE = ODE + 连续注入的高斯噪声**

---

### 4️⃣ Euler–Maruyama 方法（SDE 数值解）

$$
X_{t+h} = X_t + h u_t(X_t) + \sigma_t \sqrt{h},\varepsilon  
$$

和 Euler 的区别只有一句话：

> **多加了一项随机扰动**

---

### 5️⃣ Diffusion Model 的生成流程

1. 初始化：  
$$
    X_0 \sim p_{\text{init}}  
$$

2. 用 NN 参数化 drift：  
$$
    u^\theta_t(x)  
$$
3. 固定噪声强度 $\sigma_t$ 
4. 模拟 SDE
5. 输出：

$$
    X_1 \sim p_{\text{data}}  
$$

📌 **关键统一视角**：

> Flow model = Diffusion model 在 ( $\sigma_t$ = 0 ) 的特例

---

## 五、一个统一的认知框架（非常重要）

你可以把本课程的所有模型都压缩为一句话：

> **生成模型 = 学一个时间依赖的向量场，让噪声分布沿着 ODE / SDE 演化成数据分布**
