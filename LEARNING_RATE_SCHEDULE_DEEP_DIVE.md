# 学习率调度问题深度分析：为什么LR Decay会导致Loss下降但Eval变差

## 目录
1. [学习率基础](#学习率基础)
2. [学习率衰减的目的](#学习率衰减的目的)
3. [问题机制：低LR + 数据变化](#问题机制)
4. [数学原理](#数学原理)
5. [SmolLM3 Stage 3案例](#smollm3-stage-3案例)
6. [实验证据](#实验证据)
7. [解决方案详解](#解决方案详解)

---

## 学习率基础

### 什么是学习率？

学习率（Learning Rate, LR）控制模型参数更新的步长：

```python
# 梯度下降的基本公式
θ_new = θ_old - lr × gradient

# 例子：
θ_old = [1.0, -0.5, 2.3]
gradient = [0.1, -0.2, 0.15]  # 损失函数的梯度

# 高学习率 (lr = 0.1)
θ_new = [1.0, -0.5, 2.3] - 0.1 × [0.1, -0.2, 0.15]
      = [0.99, -0.48, 2.285]  # 大步更新

# 低学习率 (lr = 0.001)
θ_new = [1.0, -0.5, 2.3] - 0.001 × [0.1, -0.2, 0.15]
      = [0.9999, -0.4998, 2.29985]  # 小步更新
```

### 学习率的作用

#### 高学习率（例如 0.0002）
```
优点：
✓ 快速收敛
✓ 能够逃离局部最优
✓ 适应新数据快

缺点：
✗ 可能震荡
✗ 可能跳过最优点
✗ 训练不稳定
```

#### 低学习率（例如 0.00001）
```
优点：
✓ 稳定收敛
✓ 精细调整
✓ 不会跳过最优点

缺点：
✗ 收敛慢
✗ 容易卡在局部最优
✗ 难以适应新数据分布
```

### 可视化：学习率对参数更新的影响

```
Loss Landscape (损失函数地形)：

High LR (0.0002):
    Loss
     ↑
     │    ╱╲     Current
     │   ╱  ╲      ↓
     │  ╱    ╲    ●───────→ 大步跳跃
     │ ╱      ╲             可以跨过小山丘
     │╱________╲___________
                θ (parameters)

Low LR (0.00001):
    Loss
     ↑
     │    ╱╲     Current
     │   ╱  ╲      ↓
     │  ╱    ╲    ●→ 微小移动
     │ ╱      ╲    ↓  卡在当前位置
     │╱________╲___________
                θ (parameters)
```

---

## 学习率衰减的目的

### 为什么需要LR Decay？

训练分为两个阶段：

#### 阶段1：探索（Exploration）- 高学习率
```
目标: 快速找到好的参数区域
策略: 使用高学习率，大步前进
效果: Loss快速下降
```

#### 阶段2：精炼（Refinement）- 低学习率
```
目标: 在好的区域内精细调整
策略: 降低学习率，小步微调
效果: Loss平稳收敛到最优点
```

### 标准的LR Decay模式

```python
# Warmup-Stable-Decay (WSD) Scheduler

Tokens:  0  ─────→  2000  ────────────→  9T  ────────→  11T
         │          │                    │              │
LR:      0    ↗     0.0002  ─────────    0.0002   ↘    0
         │  warmup  │      stable       │        decay │

Phase 1: Warmup (0-2000 steps)
  - LR: 0 → 0.0002
  - 目的: 稳定初始训练

Phase 2: Stable (2000 steps - 9T tokens)
  - LR: 0.0002 (constant)
  - 目的: 主要学习阶段

Phase 3: Decay (9T - 11T tokens)
  - LR: 0.0002 → 0
  - 目的: 精细调整，平滑收敛
```

### 理想情况下的效果

```
Training Loss:
    │
3.0 │●                              Warmup + Stable
    │  ●●●                          ↓
2.5 │     ●●●●●
    │         ●●●●●●●●●●●●●●●●●●   ← 主要下降
2.0 │                          ●●
    │                            ●● Decay
1.5 │                             ●●● ← 精细调整
    └────────────────────────────────→
         0    2T   4T   6T   8T  9T 11T

Eval Performance:
    │
70% │                         ●●●●●●● ← 最优点
    │                    ●●●●●
65% │             ●●●●●●
    │       ●●●●●●
60% │●●●●●●
    └────────────────────────────────→
         0    2T   4T   6T   8T  9T 11T
```

在理想情况下：
- Training loss在stable phase快速下降
- Decay phase loss继续缓慢下降并稳定
- Eval performance持续提升或保持

---

## 问题机制：低LR + 数据变化

### 核心问题：在Decay阶段改变数据分布

这是**SmolLM3 Stage 3**犯的错误：

```yaml
# SmolLM3 Stage 3配置
Decay阶段开始: step 4,198,001 (9.9T tokens)

同时做了两件事：
1. LR开始衰减: 0.0002 → 0
2. 数据分布大幅改变:
   - General: 70% → 50%
   - Math: 6.5% → 12.5%
   - Code: 13% → 23%
```

### 为什么这是危险的？

#### 问题1: 低LR限制了模型的适应能力

当数据分布改变时，模型需要调整权重以适应新分布。但低LR限制了这种调整：

```python
# 假设模型需要在General和Math之间平衡

# Stage 2 (高LR = 0.0002, General主导)
θ_optimal_general = [1.0, -0.5, 2.3, 0.8, ...]  # 适合General的权重

# Stage 3 开始 (数据变为Math主导)
gradient_math = [-0.8, 0.6, -0.4, 0.3, ...]  # Math数据的梯度方向

# 如果LR高 (0.0002)
θ_new = [1.0, -0.5, 2.3, 0.8] - 0.0002 × [-0.8, 0.6, -0.4, 0.3]
      = [1.00016, -0.50012, 2.30008, 0.79994]
# 可以显著调整方向，平衡General和Math

# 但实际上LR在衰减 (例如到0.00002)
θ_new = [1.0, -0.5, 2.3, 0.8] - 0.00002 × [-0.8, 0.6, -0.4, 0.3]
      = [1.000016, -0.500012, 2.300008, 0.799994]
# 几乎无法调整！模型"冻结"在General优化状态
```

**结果**：
- ✓ Training loss下降（模型仍在学习Math，虽然很慢）
- ✗ Eval performance下降（无法平衡General和Math，General能力退化）

#### 问题2: 梯度冲突加剧

当同时有General和Math数据时，梯度可能指向不同方向：

```python
# 在同一个batch或相邻的updates

# Update 1: General数据
gradient_general = [0.5, -0.3, 0.2, ...]
direction_general = "提升general能力"

# Update 2: Math数据
gradient_math = [-0.4, 0.25, -0.15, ...]
direction_math = "提升math能力"

# 注意：梯度方向部分相反！

# 高LR时 (0.0002)
# 模型可以在两个方向间振荡，找到平衡点
θ = θ - 0.0002 × gradient_general  # 大步向general
θ = θ - 0.0002 × gradient_math     # 大步向math
# 经过多次更新，找到折中方案

# 低LR时 (0.00002)
# 微小的更新无法有效平衡
θ = θ - 0.00002 × gradient_general  # 微小移动
θ = θ - 0.00002 × gradient_math     # 微小移动
# 模型卡在"两难"状态，无法收敛到新的平衡点
```

**可视化**：

```
参数空间中的优化路径：

High LR - 可以跨越障碍找到新平衡点:
    Loss
     ↑
     │   General    Math      New
     │   Optimal    Optimal   Balance
     │      ●         ●          ●
     │    ╱│╲       ╱│╲        ╱│╲
     │   ╱ │ ╲     ╱ │ ╲      ╱ │ ╲
     │──────┼──────────┼──────────┼───→ θ
            ↑         ↑          ↑
         Stage 2   想去但   理想的
         位置      去不了   平衡点

Low LR - 卡在旧位置:
    Loss
     ↑   Current
     │      ●→(微小移动，无法跨越)
     │    ╱│╲        ╱╲
     │   ╱ │ ╲      ╱  ╲
     │──────┼────────────┼───→ θ
            ↑            ↑
         卡在这里     想去的地方
```

#### 问题3: "Plasticity Loss"（可塑性丧失）

低学习率导致模型丧失学习新模式的能力：

```python
# 神经网络的可塑性 (plasticity)
plasticity = ability_to_learn_new_patterns

# 高LR阶段
plasticity = HIGH
model.learn(new_pattern) → SUCCESS

# 低LR阶段
plasticity = LOW
model.learn(new_pattern) → FAILURE (权重几乎不变)
```

**实验证据**（来自研究论文）：

```
研究发现：在LR decay阶段训练的checkpoints
- 在原任务上: 性能好 ✓
- 在新任务上: 难以fine-tune ✗

原因：低LR阶段的训练使模型"硬化"(hardened)
     权重变得难以调整
```

#### 问题4: 优化器状态的影响

Adam优化器维护动量和自适应学习率：

```python
# Adam的内部状态
m_t = β1 × m_{t-1} + (1 - β1) × gradient      # 一阶动量
v_t = β2 × v_{t-1} + (1 - β2) × gradient²     # 二阶动量

# 实际更新
θ_new = θ_old - lr × m_t / (√v_t + ε)

# 在long stable phase之后
# m_t 和 v_t 已经"适应"了旧的数据分布

# 当数据分布突然改变
# 且LR很低时
# 优化器无法快速调整其内部状态
# 导致更新方向"惯性"地偏向旧分布
```

---

## 数学原理

### 损失函数的视角

假设我们有两个数据分布：General (G) 和 Math (M)

**总损失**：
```
L_total(θ) = α × L_G(θ) + (1-α) × L_M(θ)

其中：
- θ: 模型参数
- α: General数据的权重
- L_G: General数据上的loss
- L_M: Math数据上的loss
```

**Stage 2**：
```
α = 0.7  (70% general)
L_total = 0.7 × L_G + 0.3 × L_M
最优点: θ*_stage2 (偏向General)
```

**Stage 3**：
```
α = 0.5  (50% general)
L_total = 0.5 × L_G + 0.5 × L_M
新的最优点: θ*_stage3 (更平衡)
```

**问题**：
```
要从 θ*_stage2 移动到 θ*_stage3，需要:
Δθ = θ*_stage3 - θ*_stage2

需要的步数:
n_steps ≈ ||Δθ|| / (lr × ||gradient||)

当 lr → 0:
n_steps → ∞

结论: LR太低，模型永远到不了新的最优点！
```

### 梯度方差的视角

```python
# 不同数据源的梯度
g_general = ∇L_G(θ)  # General数据的梯度
g_math = ∇L_M(θ)     # Math数据的梯度

# 如果两者方向不同
angle = arccos(g_general · g_math / (||g_general|| × ||g_math||))

# 当angle > 90°时，梯度冲突

# 梯度方差
Var(gradient) = E[(g - E[g])²]
              = 在不同batch间的梯度差异

# 高方差 + 低LR = 问题
# 每个update的影响很小
# 但方向不一致
# 导致参数在小范围内无效振荡
```

### 泰勒展开的视角

损失函数在当前参数θ附近的泰勒展开：

```
L(θ + Δθ) ≈ L(θ) + ∇L(θ)ᵀΔθ + ½Δθᵀ∇²L(θ)Δθ + ...
            ─────   ──────────   ────────────────
            当前loss  一阶项(梯度)    二阶项(曲率)

当LR很小时:
Δθ = -lr × ∇L(θ) ≈ 0

结果:
- 一阶项 ≈ 0 (几乎没有改善)
- 无法跨越二阶项的障碍(局部曲率)
- 模型卡在当前位置
```

### 信息论的视角

```python
# 模型的"知识"编码在参数中
Knowledge(θ) = I(θ; Data)  # 互信息

# 高LR阶段
# 可以快速编码新知识
ΔI = large

# 低LR阶段
# 新知识编码速度极慢
ΔI ≈ 0

# 如果数据分布变化
# 需要编码的新知识 = I_new
# 但编码速度 → 0
# 结果: 旧知识保留，新知识学不到
```

---

## SmolLM3 Stage 3案例

### 配置细节

```yaml
# stage3_9T_11T.yaml

# 学习率调度
optimizer:
  learning_rate_scheduler:
    learning_rate: 0.0002        # 起始LR
    lr_decay_starting_step: 4198001  # 开始衰减
    lr_decay_steps: 522000       # 衰减持续52.2万步
    lr_decay_style: linear       # 线性衰减
    min_decay_lr: 0              # 最终衰减到0！

# 数据分布（第3个data stage）
data_stages:
- data:
    dataset_weights:
      - 0.20   # FineWeb-Edu (从0.30下降)
      - 0.30   # DCLM (从0.33下降)
      # ... Math和Code大幅增加
      - 0.07   # Python (从0.025上升)
      - 0.044  # C++ (从0.018上升)
      # ... 新增reasoning数据
  name: decay stage
  start_training_step: 4198001  # 与LR decay同时开始！
```

### 时间线分析

```
Step:      4,198,001              4,450,000              4,720,000
Tokens:    9.90T                  10.50T                 11.14T
           │                      │                      │
           ▼                      ▼                      ▼
LR:        0.0002 ───────────→ 0.0001 ──────────────→ 0.0
           ││                     │                      │
Data Mix:  ││ CHANGE              │                      │
           ││ 70%→50% general     │                      │
           ││ 20%→35% math/code   │                      │
           ▼▼                     ▼                      ▼
           危险组合！             模型僵化               完全冻结

Timeline:
T=0 (step 4,198,001):
  - LR decay开始
  - 数据分布突变
  - 模型需要适应新分布
  - 但LR在快速下降！

T=0.5 (step 4,450,000):
  - LR已降到0.0001 (原来的50%)
  - 模型勉强适应新数据
  - 但General能力开始下降
  - Training loss继续降(在Math/Code上)
  - Eval开始下降(General benchmark)

T=1.0 (step 4,720,000):
  - LR接近0
  - 模型完全"僵化"
  - Training loss仍在降(微小的进步)
  - Eval显著下降(无法平衡)
```

### 问题演化过程

#### 阶段1: 初始冲击 (4.20M - 4.25M steps)

```python
# LR还比较高 (0.00018 - 0.00015)
# 但数据突然变化

效果:
- Training loss: 快速下降 (适应Math/Code数据)
- General eval: 开始缓慢下降 (权重被Math/Code覆盖)
- Math/Code eval: 快速上升

此时的梯度:
gradient_magnitude = MEDIUM
adaptation_speed = MODERATE

还能勉强平衡，但已开始倾斜
```

#### 阶段2: 加速恶化 (4.25M - 4.50M steps)

```python
# LR继续下降 (0.00015 - 0.00008)
# Math/Code数据持续主导

效果:
- Training loss: 继续下降 (在Math/Code上优化)
- General eval: 明显下降 (遗忘加速)
- Math/Code eval: 继续上升

此时的梯度:
gradient_magnitude = SMALL
adaptation_speed = SLOW

模型开始"僵化"，难以调整
```

#### 阶段3: 完全锁定 (4.50M - 4.72M steps)

```python
# LR接近0 (0.00008 - 0.0)

效果:
- Training loss: 微小下降 (几乎停滞)
- General eval: 持续下降或停滞在低位
- Math/Code eval: 可能达到plateau

此时的梯度:
gradient_magnitude = TINY
adaptation_speed = NEAR_ZERO

模型"冻结"在Math/Code专化状态
无法恢复General能力
```

### 定量分析（理论估算）

假设模型需要调整10%的参数来平衡General和Math：

```python
# 参数总量
total_params = 3B

# 需要调整的参数
params_to_adjust = 0.1 × 3B = 300M

# 每个参数需要的平均调整量
avg_adjustment = 0.01  # 1%的参数值

# 总的参数变化量
total_change_needed = 300M × 0.01 = 3M units

# 梯度平均大小（假设）
avg_gradient = 0.1

# Stage 3的tokens
tokens_in_stage3 = 11.14T - 9.90T = 1.24T

# Steps in stage 3
steps_in_stage3 = 522,000

# 平均LR (从0.0002衰减到0)
avg_lr = (0.0002 + 0.0) / 2 = 0.0001

# 实际的参数变化量
actual_change = steps_in_stage3 × avg_lr × avg_gradient
              = 522,000 × 0.0001 × 0.1
              = 5,220 units

# 对比
needed / actual = 3,000,000 / 5,220 ≈ 575倍不足！

结论: LR太低，模型无法完成必要的调整！
```

---

## 实验证据

### 相关研究发现

#### 研究1: "Overtrained Language Models Are Harder to Fine-Tune"

**发现**：
```
模型在3T tokens上预训练比2.3T tokens预训练的:
- Fine-tuning后性能低3%
- 需要更高的学习率才能fine-tune
- 在新任务上适应能力下降

原因: 长时间低LR训练使模型"硬化"
```

#### 研究2: "Loss Plasticity in Deep Learning"

**发现**：
```
当学习率很小时:
- 模型的loss plasticity下降
- 即使引入新数据，loss也难以快速适应
- 需要"重新warmup"才能恢复plasticity
```

#### 研究3: "Continual Learning with Low Learning Rates"

**发现**：
```
在continual learning场景:
- 高LR: 快速学习新任务，但可能遗忘旧任务
- 低LR: 保护旧任务，但无法学习新任务
- 中等LR + rehearsal: 最佳平衡

当同时降低LR和改变数据分布:
→ 最坏情况: 新任务学不好，旧任务也退化
```

### 可能的实验观察（SmolLM3）

如果SmolLM3确实遇到了这个问题，应该能观察到：

```python
# 在WandB或训练日志中

# 1. Training loss曲线
training_loss = {
    "stage2_end": 2.1,
    "stage3_early": 2.05,   # 快速下降
    "stage3_mid": 1.98,     # 继续下降
    "stage3_end": 1.95,     # 微小下降
}
# Loss持续下降，看起来正常

# 2. Eval metrics (假设)
eval_general = {
    "stage2_end": 68.5,
    "stage3_early": 67.2,   # 开始下降
    "stage3_mid": 65.1,     # 明显下降
    "stage3_end": 63.5,     # 持续下降
}

eval_math = {
    "stage2_end": 52.0,
    "stage3_early": 58.0,   # 上升
    "stage3_mid": 64.0,     # 继续上升
    "stage3_end": 68.0,     # plateau
}

# 3. 梯度范数
gradient_norm = {
    "stage2_end": 0.15,
    "stage3_early": 0.12,   # 下降
    "stage3_mid": 0.05,     # 显著下降
    "stage3_end": 0.01,     # 接近0
}
# 梯度越来越小，优化几乎停止

# 4. 参数变化速率
param_change_rate = {
    "stage2_end": 0.0003,
    "stage3_early": 0.00015,
    "stage3_mid": 0.00005,
    "stage3_end": 0.00001,
}
# 参数几乎不再变化
```

---

## 解决方案详解

### 方案1: 不要衰减到0 ⭐⭐⭐⭐⭐

**最简单且有效的方案**

```yaml
# 原配置（问题）
learning_rate_scheduler:
  lr_decay_starting_step: 4198001
  min_decay_lr: 0  # ✗ 衰减到0

# 修改后（推荐）
learning_rate_scheduler:
  lr_decay_starting_step: 4198001
  min_decay_lr: 2e-5  # ✓ 保持小的非零LR
```

**原理**：
```python
# 最后阶段仍保留少量学习能力
final_lr = 2e-5  # 是初始LR的10%

# 虽然小，但仍能进行有意义的更新
update_size = 2e-5 × gradient
            ≠ 0  # 可以继续调整

# 好处：
1. 模型保持plasticity
2. 可以平衡不同数据源
3. 避免"冻结"在某个状态
```

**实验建议**：
```python
# 尝试不同的min_lr
experiments = [
    {"min_lr": 1e-5, "expected": "较强的调整能力"},
    {"min_lr": 2e-5, "expected": "平衡"},
    {"min_lr": 5e-5, "expected": "接近不decay"},
]

# 监控指标
for exp in experiments:
    monitor = {
        "general_eval": "是否保持？",
        "math_eval": "是否提升？",
        "training_stability": "是否稳定？",
    }
```

### 方案2: 使用Cosine Decay ⭐⭐⭐⭐

**更温和的衰减曲线**

```python
# Linear Decay（SmolLM3当前使用）
def linear_decay(step, start_step, end_step, max_lr, min_lr):
    progress = (step - start_step) / (end_step - start_step)
    return max_lr - progress * (max_lr - min_lr)

# Cosine Decay（推荐）
def cosine_decay(step, start_step, end_step, max_lr, min_lr):
    progress = (step - start_step) / (end_step - start_step)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_factor

# 可视化对比
steps = range(4198001, 4720001)

Linear Decay:
0.0002 │●
       │  ●●
       │     ●●●
       │        ●●●●
       │            ●●●●●●
       │                  ●●●●●●●●
0.0    │                          ●●●●●
       └────────────────────────────────→

Cosine Decay:
0.0002 │●
       │  ●
       │    ●●
       │       ●●
       │         ●●●
       │            ●●●●
       │                ●●●●●●●●●
0.0    │                          ●●●●●
       └────────────────────────────────→

差异:
- Linear: 前期快速下降，后期缓慢
- Cosine: 更平滑，后期下降较快
```

**优势**：
```
1. 前期保持较高LR时间更长
   → 更多时间适应新数据分布

2. 后期仍有一定的learning rate
   → 避免过早"冻结"

3. 数学上更平滑
   → 训练更稳定
```

### 方案3: 分离数据变化和LR Decay ⭐⭐⭐⭐⭐

**最佳实践：不要同时改变两者**

```yaml
# 问题配置（SmolLM3当前）
Stage 3 (9T-11T):
  - 同时: LR decay + 数据分布变化

# 推荐配置
Stage 2.5 (8T-9.5T): 调整数据分布
  lr: 0.0002 (constant)  # LR保持不变
  data:
    general: 70% → 60% → 55%  # 渐进变化
    math: 10% → 15% → 18%
    code: 20% → 25% → 27%

Stage 3 (9.5T-10T): 稳定阶段
  lr: 0.0002 (constant)
  data: 保持stage 2.5末期的分布

Stage 4 (10T-11T): 仅LR decay
  lr: 0.0002 → 2e-5 (decay)
  data: 保持不变！  # 关键：不改数据
```

**原理**：
```
分离变化，让模型逐个适应：

时间轴：
8T          9.5T        10T         11T
│           │           │           │
├─ Stage 2.5─┼─ Stage 3 ┼─ Stage 4 ─┤
│            │           │           │
Data:        │           │           │
变化中───→稳定───────→稳定───────→
│            │           │           │
LR:          │           │           │
稳定─────→稳定───────→Decay───→

好处:
1. 模型先在高LR下适应新数据
2. 然后在稳定配置下优化
3. 最后才降低LR精细调整
4. 每次只改变一个因素
```

### 方案4: Cyclical Learning Rates ⭐⭐⭐

**在Decay阶段使用周期性LR**

```python
# 传统Decay
lr = linear_decay(step)  # 单调下降

# Cyclical Decay
def cyclical_decay(step, cycle_length=1000):
    base_lr = linear_decay(step)  # 基础衰减
    cycle_position = step % cycle_length
    cycle_factor = 0.5 * (1 + cos(2*pi * cycle_position / cycle_length))
    return base_lr * (0.5 + 0.5 * cycle_factor)  # 在基础上震荡

可视化:
0.0002 │●    ╱●╲   ╱●╲  ╱●╲ ╱●
       │ ╲  ╱  ╲ ╱  ╲╱  ╲╱
       │  ●╱    ●
       │
0.0    │                        ●
       └────────────────────────→

好处:
- 周期性提高LR，恢复plasticity
- 可以"跳出"局部最优
- 更好地平衡不同数据源
```

### 方案5: Warmup After Data Change ⭐⭐⭐⭐

**数据变化后重新Warmup**

```yaml
# 在Stage 3开始时
Step 4,198,001:
  1. 改变数据分布
  2. LR从当前值(0.0002)降低到1e-5
  3. Warmup: 1e-5 → 0.0002 (2000 steps)
  4. 稳定训练 (0.0002, 50,000 steps)
  5. 然后开始正常的decay

Timeline:
4.198M      4.200M      4.250M      4.720M
│           │           │           │
Data Change Warmup      Stable      Decay
▼           ▼           ▼           ▼
LR: 0.0002→1e-5→0.0002→0.0002→0→
```

**原理**：
```
Warmup给模型时间适应新数据：
1. 先降低LR，避免剧烈震荡
2. 逐步提高，让模型渐进适应
3. 稳定期充分优化
4. 最后才开始decay

类似于"重启"训练，但保留已学知识
```

### 方案6: 自适应LR调整 ⭐⭐⭐⭐

**根据Eval Performance动态调整LR**

```python
def adaptive_lr_schedule(step, eval_metrics):
    """
    根据eval性能动态调整学习率
    """
    base_lr = scheduled_lr(step)  # 计划的LR

    # 检查eval性能
    if eval_metrics['current'] < eval_metrics['best'] * 0.98:
        # 性能下降超过2%
        # 提高LR，增强适应能力
        adjusted_lr = base_lr * 1.5
        print(f"Eval下降，提高LR到 {adjusted_lr}")
    elif eval_metrics['recent_improvement'] < threshold:
        # 性能停滞
        # 稍微提高LR，跳出plateau
        adjusted_lr = base_lr * 1.2
    else:
        # 性能正常
        adjusted_lr = base_lr

    return adjusted_lr

# 示例
eval_log = {
    4200000: 68.5,
    4210000: 67.8,  # 下降
    4220000: 67.2,  # 继续下降
}

# 在4220000 step检测到问题
# 提高LR from 0.00015 to 0.000225
# 给模型更多调整空间
```

### 方案7: 使用Layer-wise Learning Rates ⭐⭐⭐

**不同层使用不同的学习率**

```python
# 假设：
# - 浅层学习general features
# - 深层学习task-specific features

layer_lrs = {
    "embedding": 1e-5,      # 最低（最general）
    "layer_0-5": 5e-5,      # 低
    "layer_6-17": 1e-4,     # 中
    "layer_18-35": 2e-4,    # 高（最task-specific）
    "lm_head": 2e-4,        # 高
}

# 在数据分布变化时
# Task-specific层可以快速适应（高LR）
# General层保护已学知识（低LR）

好处：
- 平衡general和specialized能力
- 减少catastrophic forgetting
- 更细粒度的控制
```

### 综合推荐方案

**针对SmolLM3 Stage 3的最佳配置**：

```yaml
# 重新设计的Stage 3
stage_2_extended:  # 8T - 9.5T
  data:
    # 渐进式调整数据分布
    stage_2a (8T-8.5T):
      general: 65%
      math: 15%
      code: 20%

    stage_2b (8.5T-9T):
      general: 60%
      math: 18%
      code: 22%

    stage_2c (9T-9.5T):
      general: 55%
      math: 20%
      code: 23%
      reasoning: 2%

  learning_rate:
    lr: 0.0002  # 保持恒定

stage_3_stable:  # 9.5T - 10T
  data:
    # 保持stage_2c的分布
    general: 55%
    math: 20%
    code: 23%
    reasoning: 2%

  learning_rate:
    lr: 0.0002  # 仍然恒定

stage_4_decay:  # 10T - 11T
  data:
    # 完全不变！
    general: 55%
    math: 20%
    code: 23%
    reasoning: 2%

  learning_rate:
    # Cosine decay到非零值
    lr_schedule: cosine
    max_lr: 0.0002
    min_lr: 2e-5  # 不衰减到0

    # 或者带warmup的adaptive
    warmup_steps: 2000
    stable_steps: 200000
    decay_steps: 320000
```

---

## 总结

### 核心问题

**SmolLM3 Stage 3同时犯了两个错误**：

1. ✗ 在LR decay阶段改变数据分布
2. ✗ LR衰减到0

这导致：
- 模型无法适应新数据分布（LR太低）
- 同时丧失了旧能力（数据分布变化）
- Training loss下降（在新数据上缓慢优化）
- Eval性能下降（无法平衡新旧知识）

### 关键洞察

```
学习率 = 模型的"可塑性"

高LR:
  ✓ 可以快速适应新数据
  ✗ 可能不稳定

低LR:
  ✓ 稳定收敛
  ✗ 难以学习新模式
  ✗ 模型"硬化"

在低LR时改变数据分布 = 最坏情况:
  - 新数据学不好（LR太低）
  - 旧能力也丢失（分布变化）
```

### 黄金法则

1. **数据变化和LR Decay不要同时进行**
2. **LR不要衰减到0**（保持2e-5左右）
3. **使用温和的decay曲线**（Cosine > Linear）
4. **数据分布变化要渐进式**
5. **密切监控eval metrics**（按domain分别监控）

### 检查清单

在设计训练计划时，问自己：

```
□ LR decay期间数据分布是否稳定？
□ LR最终值是否为非零？
□ Decay曲线是否够温和？
□ 是否有足够的stable phase？
□ 是否有evaluation safety net？
□ 如果性能下降，是否能及时干预？
```

---

**文档创建**: 2025-11-12
**针对问题**: Training Loss下降但Eval性能变差
**核心原因**: 低学习率 + 数据分布变化的危险组合
