
# 0️⃣ del 算子（∇ / nabla）

**定义**  
$$
\nabla
\;:=\;
\left(
\frac{\partial}{\partial x_1},
\frac{\partial}{\partial x_2},
\ldots,
\frac{\partial}{\partial x_n}
\right)
$$

**本质**  
- ∇ **不是一个具体运算**，而是一个**一阶微分算子向量**
- 它本身没有意义，**必须与场“组合”**才产生运算
- 可以看作：
  > “把方向导数按坐标方向打包成一个对象”

---

# 1️⃣ 梯度（Gradient）

**对象**：标量场  
$$
f:\mathbb{R}^n \to \mathbb{R}
$$

**公式**  
$$
\nabla f
=
\left(
\frac{\partial f}{\partial x_1},
\ldots,
\frac{\partial f}{\partial x_n}
\right)
$$

**几何意义**  
- 指向 **函数增长最快的方向**  
- 模长 = **最大方向导数**
- 与等值面（level set）**正交**

**本质**  
- 是 **微分 $df$** 在内积结构下对应的向量  
- 把“标量变化率”转成一个**方向性对象**  
- 属于：  
$$
\text{Gradient}:\ \text{scalar} \;\to\; \text{vector}
$$

---

# 2️⃣ 散度（Divergence）

**对象**：向量场  
$$
\mathbf{v}:\mathbb{R}^n \to \mathbb{R}^n
$$

**公式**  
$$
\nabla \cdot \mathbf{v}
=
\sum_{i=1}^n
\frac{\partial v_i}{\partial x_i}
$$

**几何意义**  
- 描述一点处是否像**源 / 汇**
- 正：向外“发散”  
- 负：向内“汇聚”
- 等价于**单位体积的净流出量**

**本质**  
- 是向量场的 **局部体积变化率**
- 是通量定理（Gauss 定理）的**微分形式**
- 属于：  
$$
\text{Divergence}:\ \text{vector} \;\to\; \text{scalar}
$$


---

# 3️⃣ 旋度（Curl）【3D】

**对象**：向量场  
$$
\mathbf{v}=(v_1,v_2,v_3)
$$

**公式**  
$$
\nabla \times \mathbf{v}
=
\begin{vmatrix}
\mathbf{i}&\mathbf{j}&\mathbf{k}\\
\partial_x&\partial_y&\partial_z\\
v_1&v_2&v_3
\end{vmatrix}
$$

**几何意义**  
- 描述局部**旋转趋势**
- 方向：右手法则（旋转轴）
- 大小：单位面积上的**环流强度**

**本质**  
- 是向量场的 **局部反对称变化**
- 对应 Stokes 定理的**微分形式**
- 属于：  
$$
\text{Curl}:\ \text{vector} \;\to\; \text{vector}
$$


---

# 4️⃣ 一句话对比（记忆用）

| 运算     | 输入  | 输出  | 核心本质    |
| ------ | --- | --- | ------- |
| 梯度 ∇f  | 标量场 | 向量  | 最陡上升方向  |
| 散度 ∇·v | 向量场 | 标量  | 局部源 / 汇 |
| 旋度 ∇×v | 向量场 | 向量  | 局部旋转    |

---
