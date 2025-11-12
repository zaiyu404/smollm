# Training Loss下降但Evaluation变差：原因分析

## 问题定义

**现象描述**：
- ✓ Training Loss持续下降（看似训练进展良好）
- ✗ Evaluation Metrics下降（实际模型能力变差）

这是比loss spike更严重的问题，表明模型的**泛化能力正在恶化**。

---

## 核心原因分类

### 1. 经典过拟合（Classic Overfitting）

#### 定义
模型过度记忆训练数据的特定模式，丧失泛化能力。

#### 在LLM中的表现
```
Training Data: "The capital of France is Paris"
模型记忆: 精确匹配这个句子 → training loss ↓

Evaluation Data: "What is the capital city of France?"
模型失败: 无法泛化到不同的表述方式 → eval performance ↓
```

#### 触发条件
- **训练时间过长**：超过最优点继续训练
- **数据量不足**：相对模型容量，数据太少
- **数据重复度高**：多个epoch导致记忆

#### 检测方法
```python
# 监控指标
training_loss_curve = [2.5, 2.3, 2.1, 1.9, 1.7, 1.5]  # 持续下降
validation_loss_curve = [2.6, 2.4, 2.3, 2.4, 2.6, 2.9]  # 先降后升
eval_metrics = [65, 67, 68, 66, 63, 60]  # 性能下降

# 典型模式：validation loss拐点
if val_loss[t] > val_loss[t-1] and train_loss[t] < train_loss[t-1]:
    print("过拟合开始")
```

#### 解决方案
- **Early Stopping**：在validation loss开始上升时停止
- **正则化**：L2 weight decay, dropout
- **数据增强**：增加训练数据多样性
- **减少模型容量**：使用更小的模型

---

### 2. 数据分布偏移（Distribution Shift）

#### 定义
训练数据分布与评测数据分布不匹配，模型在训练分布上优化但在评测分布上性能下降。

#### 在LLM训练中的表现

**Scenario A: 训练阶段数据变化**
```yaml
# Stage 1-2: 通用网页数据为主
Training Data: 70% general web, 15% code, 15% math
Eval Benchmarks: 主要是通用推理任务 (HellaSwag, ARC, MMLU)
Result: ✓ Training loss和eval metrics一致下降

# Stage 3: 数据分布突变
Training Data: 50% general web, 25% code, 25% math
模型优化方向: 专注于code和math
Result:
  ✓ Training loss下降 (在新数据分布上学习)
  ✗ Eval metrics下降 (通用benchmark不再匹配训练分布)
```

**Scenario B: 评测分布不匹配**
```python
# 训练数据: 主要是英文学术文本
Training: "The photosynthesis process involves..."

# 评测任务: 常识推理和日常对话
Eval: "If you drop a glass, what happens?"
      → 模型在学术文本上优化，日常推理能力下降
```

#### SmolLM3 Stage 3的具体案例

```yaml
# Stage 2 结束时 (9.9T tokens)
数据分布:
  - General Web: 70%
  - Code: 13%
  - Math: 6.5%

Eval Performance: 通用任务表现良好

# Stage 3 (9.9T → 11T)
数据分布:
  - General Web: 50% (-29%)
  - Code: 23% (+77%)
  - Math: 12.5% (+92%)
  - Reasoning Synthetic: 2.2% (新增)

Training Loss: 持续下降 (在code/math数据上学习良好)
Eval Performance: 下降 (通用benchmark不再是优化目标)
```

#### 数学原理

训练优化的目标：
```
L_train = E_{x~P_train}[loss(x)]  # 最小化训练分布的loss
```

但评测的是：
```
L_eval = E_{x~P_eval}[loss(x)]    # 评测分布的loss
```

当 `P_train ≠ P_eval` 时：
- `L_train ↓` 不保证 `L_eval ↓`
- 如果分布差异大，可能 `L_eval ↑`

#### 检测方法

**A. 分域评测**
```python
eval_by_domain = {
    "general": [65, 67, 68, 66, 63, 60],  # 下降
    "math": [45, 48, 52, 58, 65, 70],     # 上升
    "code": [40, 43, 47, 55, 62, 68],     # 上升
}

# 如果不同domain趋势相反 → 数据分布偏移
```

**B. 计算分布相似度**
```python
from scipy.stats import entropy

# KL散度：衡量两个分布的差异
kl_div = entropy(P_train, P_eval)

if kl_div > threshold:
    print("训练和评测分布差异过大")
```

#### 解决方案

**短期**：
- 使用更早的checkpoint（分布偏移前）
- Model merging：融合不同阶段的checkpoint

**长期**：
- 保持训练数据分布与评测目标一致
- 渐进式数据分布变化，避免突变
- 在stable phase做数据调整，不在decay phase

---

### 3. 灾难性遗忘（Catastrophic Forgetting）

#### 定义
在学习新知识时，模型忘记之前学到的知识。

#### 在LLM连续训练中的表现

```
Time 0-8T tokens: 学习通用知识
  模型掌握: 常识推理、语言理解、世界知识
  Eval: 通用benchmark表现好

Time 8T-10T tokens: 大量math/code数据
  模型优化: 数学和编程能力
  副作用: 覆盖之前的通用知识权重

Result:
  ✓ Math/Code能力提升
  ✗ 通用能力下降 (catastrophic forgetting)
```

#### 神经网络机制

**权重覆盖问题**：
```python
# 阶段1：学习通用知识
W_general = [0.5, 0.3, -0.2, 0.8, ...]  # 权重编码通用知识

# 阶段2：学习math/code
# 梯度更新会修改相同的权重
W_new = W_general - lr * grad_math_code
      = [0.2, 0.7, 0.1, -0.3, ...]  # 通用知识被覆盖

# 结果：
math_performance ↑
general_performance ↓  # 灾难性遗忘
```

#### 何时最容易发生

**高风险场景**：
1. **学习率衰减期间改变数据分布**
   ```yaml
   # SmolLM3 Stage 3
   start_step: 4198001
   lr_decay: 0.0002 → 0 (linear)
   data_shift: 激进的分布变化

   风险: 低学习率 + 新数据 = 选择性遗忘
   ```

2. **新数据占比过高**
   ```python
   # 如果新数据占50%+
   每个batch: 一半是新领域数据
   梯度方向: 主要由新数据主导
   结果: 旧知识权重被快速覆盖
   ```

3. **缺少旧数据的rehearsal**
   ```python
   # 完全切换到新数据
   Stage 2: [general: 70%, math: 10%, code: 20%]
   Stage 3: [general: 30%, math: 40%, code: 30%]

   # general数据从70%→30%
   # 模型在general数据上的更新频率大幅降低
   # → 遗忘加速
   ```

#### 定量检测

**Backward Transfer指标**：
```python
# 测量在旧任务上的性能变化
backward_transfer = performance_task_A_after - performance_task_A_before

# 例如：
general_before_stage3 = 68.5
general_after_stage3 = 62.3
backward_transfer = 62.3 - 68.5 = -6.2  # 负值 = 遗忘
```

**Forgetting Rate**：
```python
forgetting_rate = (best_performance - current_performance) / best_performance

# 例如：
best_general = 68.5 (at step 4,198,000)
current_general = 62.3 (at step 4,500,000)
forgetting_rate = (68.5 - 62.3) / 68.5 = 9.1%
```

#### 解决方案

**A. Experience Replay / Rehearsal**
```yaml
# 保持旧数据的足够比例
Stage 3 配置修改:
  general_web: 0.50  # 不要低于50%
  math: 0.25
  code: 0.25
```

**B. Elastic Weight Consolidation (EWC)**
```python
# 保护重要权重不被修改
loss_new = loss_task_new + λ * Σ(F_i * (θ_i - θ_i*)^2)
#                               ↑        ↑      ↑
#                           重要性   当前权重  旧权重
```

**C. Progressive Neural Networks**
```
添加新的网络分支而不修改旧分支
[General Branch] ← frozen
[Math Branch] ← new, trainable
[Code Branch] ← new, trainable
```

**D. 温和的学习率调度**
```yaml
# 不要衰减到0
learning_rate_scheduler:
  lr_decay_starting_step: 4198001
  min_decay_lr: 2e-5  # 保持小的学习率，而非0
```

---

### 4. 训练数据质量问题

#### 定义
训练数据包含噪声、偏差或不相关内容，模型学习到错误模式。

#### 在LLM训练中的表现

**Scenario A: 合成数据的artifacts**
```python
# Stage 3新增的reasoning数据（2.2%）
Synthetic Data: "Let's think step by step... <reasoning>... Therefore..."

问题:
1. 模板化的推理模式 (artifacts)
2. 不自然的语言结构
3. 偏向特定推理格式

结果:
✓ Training loss ↓ (学习模板)
✗ Eval ↓ (真实推理能力未提升，反而学到了artifacts)
```

**Scenario B: 数据污染**
```python
# 如果训练数据包含benchmark泄漏
Training Data: 包含评测题目的变体

初期: Eval performance ↑ (记忆题目)
后期: 继续训练后过拟合特定表述
     Eval performance ↓ (泛化能力下降)
```

**Scenario C: 数据质量下降**
```yaml
# 如果Stage 3引入了低质量数据
Stage 2: 高质量筛选的数据
  FineWeb-Edu (quality score > 3)
  DCLM (carefully filtered)

Stage 3: 添加大量未筛选数据
  - 某些math数据质量参差不齐
  - 某些synthetic数据包含错误

Result:
  Training loss仍下降 (模型"学习"了噪声)
  Eval下降 (学到错误模式)
```

#### 检测方法

**A. 数据审计**
```python
# 检查新加数据的质量
for dataset in new_datasets:
    # 1. 人工抽样检查
    samples = random_sample(dataset, n=100)
    quality_score = human_review(samples)

    # 2. 自动质量评估
    perplexity = calculate_perplexity(samples, reference_model)
    diversity = calculate_diversity(samples)

    if quality_score < threshold or perplexity > threshold:
        print(f"Dataset {dataset} 质量可疑")
```

**B. Ablation Study**
```python
# 移除每个新数据源，观察影响
configs = [
    "baseline + InfiMM",
    "baseline + Stack-Edu",
    "baseline + Reasoning",
    "baseline + all"
]

# 找出导致问题的数据源
```

#### 解决方案

**短期**：
- 移除可疑数据源
- 使用添加问题数据前的checkpoint

**长期**：
- 严格的数据质量控制流程
- 渐进式引入新数据，监控影响
- 对合成数据进行质量过滤

---

### 5. 学习率调度问题

#### 定义
学习率调度不当导致模型陷入次优解或过度专化。

#### 在LLM训练中的表现

**Scenario A: 学习率衰减过快**
```python
# SmolLM3 Stage 3配置
lr_schedule:
  start: 0.0002
  end: 0.0
  decay: linear
  steps: 522,000

问题:
- 学习率快速下降到接近0
- 模型"冻结"在特定的数据分布上
- 无法平衡多个数据源的学习

Time:  [Stage 3 开始] -----> [Stage 3 结束]
LR:    0.0002         -----> 0.0
Data:  通用为主       -----> Math/Code为主

结果:
- 早期(高LR): 模型快速适应新数据
- 后期(低LR): 只能微调math/code，通用能力"冻结"在下降状态
```

**Scenario B: 衰减阶段数据变化**
```yaml
传统做法（推荐）:
  Stable Phase: 调整数据分布
  Decay Phase: 保持数据分布，仅降低LR

SmolLM3实际（高风险）:
  Decay Phase开始: 同时改变数据分布 + 降低LR

风险:
  低LR → 模型僵化 → 难以平衡新旧知识
```

#### 数学直觉

```python
# 参数更新公式
θ_new = θ_old - lr * gradient

# 高学习率时期
lr = 0.0002
gradient_general = [0.1, -0.2, 0.3]
gradient_math = [-0.1, 0.15, -0.05]
# 模型可以平衡两个方向的更新

# 低学习率时期
lr = 0.00001  # 接近0
# 微小的更新无法修正已经偏离的方向
# 模型"锁定"在math/code特化状态
```

#### 检测方法

**A. 监控学习率与性能关系**
```python
checkpoints = {
    "step_4198000": {"lr": 0.0002, "eval": 68.5},  # Decay开始
    "step_4300000": {"lr": 0.00015, "eval": 67.2},
    "step_4450000": {"lr": 0.00008, "eval": 65.1},
    "step_4600000": {"lr": 0.00002, "eval": 63.5},
    "step_4720000": {"lr": 0.0, "eval": 62.3},
}

# 如果eval随lr下降而单调下降 → LR schedule问题
```

**B. 尝试不同的LR schedule**
```python
experiments = [
    {"name": "linear_to_zero", "min_lr": 0.0},
    {"name": "linear_to_small", "min_lr": 2e-5},
    {"name": "cosine", "min_lr": 1e-5},
    {"name": "constant", "min_lr": 2e-4},
]
```

#### 解决方案

**A. 不要衰减到0**
```yaml
learning_rate_scheduler:
  lr_decay_starting_step: 4198001
  min_decay_lr: 2e-5  # 保持小的LR，而非0
```

**B. 使用Cosine Decay**
```python
# 比linear更温和
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t / T))
```

**C. Warmup-Stable-Decay但在Decay前稳定数据**
```yaml
Phase 1 (0-8T): Warmup + Stable
Phase 2 (8T-9.5T): 调整数据分布，保持LR
Phase 3 (9.5T-10T): 稳定数据和LR
Phase 4 (10T-11T): 仅Decay LR，不改数据
```

---

### 6. Evaluation Benchmark不匹配

#### 定义
评测任务与训练目标不一致，导致训练有效但评测显示性能下降。

#### 在LLM训练中的表现

**Scenario: Math/Code特化 vs 通用评测**
```python
Training Objective:
  优化math和code性能
  Math data: 6.5% → 12.5%
  Code data: 13% → 23%

Evaluation Benchmarks:
  HellaSwag: 常识推理 (general)
  ARC: 科学推理 (general)
  MMLU: 知识问答 (general)
  GSM8K: 数学 (specialized)

Observation:
  HellaSwag: 68 → 63 ↓  (不是训练重点)
  ARC: 65 → 61 ↓        (不是训练重点)
  GSM8K: 52 → 68 ↑      (训练重点)

Interpretation:
  不是模型变差了，而是"方向"改变了
```

#### 真实的Trade-off

```
模型容量有限:
  ┌─────────────────────────┐
  │ General  │ Math │ Code  │  ← Total Capacity
  └─────────────────────────┘

Stage 2:
  ████████ ███ ████           ← 通用为主

Stage 3:
  █████ ███████ ██████         ← Math/Code增加，General被压缩
```

#### 是否是真正的问题？

**需要判断**：
```python
# 问题1: 训练目标是什么？
if goal == "通用模型":
    general_performance_drop = PROBLEM
elif goal == "Math/Code专家模型":
    general_performance_drop = ACCEPTABLE_TRADEOFF

# 问题2: Trade-off是否合理？
if math_gain > general_loss:
    overall_utility_increased = True
else:
    overall_utility_decreased = True
```

#### 检测方法

**全面评测**：
```python
eval_results = {
    # 通用任务
    "HellaSwag": 63,  # ↓
    "ARC": 61,        # ↓
    "MMLU": 59,       # ↓

    # 专项任务
    "GSM8K": 68,      # ↑
    "MATH": 32,       # ↑
    "HumanEval": 45,  # ↑

    # 综合判断
    "average_all": 61.3  # 如果这个上升 → 整体进步
}
```

#### 解决方案

**明确训练目标**：
```yaml
# Option A: 通用模型
optimization_goal: "balanced performance"
data_mixture: 保持通用数据主导

# Option B: 专家模型
optimization_goal: "math and code specialist"
data_mixture: 可以牺牲通用性能
```

**Multi-task Balancing**：
```python
# 使用加权loss平衡不同目标
loss = w_general * loss_general + w_math * loss_math + w_code * loss_code

# 根据目标调整权重
if goal == "balanced":
    weights = [0.7, 0.15, 0.15]
elif goal == "math_focus":
    weights = [0.5, 0.3, 0.2]
```

---

### 7. 批次效应和采样问题

#### 定义
数据采样策略导致训练批次不具代表性。

#### 在LLM训练中的表现

**Scenario A: 不平衡采样**
```python
# 数据集大小差异巨大
dataset_sizes = {
    "FineWeb": 10TB,
    "DCLM": 8TB,
    "InfiMM-WebMath": 500GB,  # 小得多
    "Reasoning": 100GB,        # 更小
}

# 如果使用简单的uniform sampling
for batch in training:
    # 大多数batch来自FineWeb/DCLM
    # InfiMM/Reasoning很少被采样

# 为了增加小数据集的影响，提高权重
weights = {
    "FineWeb": 0.2,
    "InfiMM": 0.02,  # 实际采样比例远超数据量占比
}

问题:
- 小数据集被过度采样（多个epoch）
- 导致对这些数据过拟合
- Training loss ↓ (记忆小数据集)
- Eval ↓ (过拟合降低泛化)
```

**Scenario B: 批次内分布不均**
```python
# 如果批次构建不当
batch_1 = [math_sample] * 1000  # 全是math
batch_2 = [general_sample] * 1000  # 全是general

问题:
- 梯度方差大
- 训练不稳定
- 可能导致某些能力被周期性地"遗忘"和"重学"
```

#### 检测方法

**A. 数据统计**
```python
# 检查每个数据源被采样的次数
sample_counts = count_samples_per_epoch(training_log)

for dataset, count in sample_counts.items():
    epoch_count = count / dataset_size[dataset]
    if epoch_count > 2.0:
        print(f"{dataset} 过度采样: {epoch_count:.1f} epochs")
```

**B. 批次分析**
```python
# 分析批次的domain分布
for batch in sample_batches(training_log, n=100):
    domain_dist = count_domains(batch)
    entropy_score = calculate_entropy(domain_dist)

    if entropy_score < threshold:
        print("批次多样性不足")
```

#### 解决方案

**A. 温度采样（Temperature Sampling）**
```python
# 平滑采样概率
def get_sampling_prob(dataset_size, temperature=0.5):
    # temperature=1.0: 按数据量采样
    # temperature=0.0: 均匀采样
    # temperature=0.5: 折中
    prob = dataset_size ** temperature
    return prob / sum(all_probs)
```

**B. Batch-level Mixing**
```python
# 每个batch包含多个domain的样本
batch = []
for domain in domains:
    n_samples = int(batch_size * domain_weight[domain])
    batch.extend(sample(domain_data, n_samples))
random.shuffle(batch)
```

**C. 限制Epoch数量**
```python
# 对小数据集限制重复次数
max_epochs = {
    "large_dataset": inf,
    "small_dataset": 2,  # 最多2个epoch
}
```

---

## 综合诊断流程

### Step 1: 确定问题类型

```python
def diagnose_loss_eval_divergence(training_log, eval_log):
    """
    诊断loss下降但eval变差的具体原因
    """

    # 1. 检查validation loss
    if validation_loss_increasing():
        return "CLASSIC_OVERFITTING"

    # 2. 检查不同domain的表现
    domain_trends = analyze_domain_performance()
    if some_domains_improve and some_domains_degrade:
        return "DISTRIBUTION_SHIFT"

    # 3. 检查forgetting
    backward_transfer = calculate_backward_transfer()
    if backward_transfer < -0.05:  # 下降超过5%
        return "CATASTROPHIC_FORGETTING"

    # 4. 检查数据变化时间点
    if data_mixture_changed_recently():
        return "DATA_DISTRIBUTION_CHANGE"

    # 5. 检查学习率
    if learning_rate_near_zero():
        return "LR_SCHEDULE_ISSUE"

    # 6. 检查数据质量
    if new_data_added_recently():
        return "DATA_QUALITY_ISSUE"

    return "UNKNOWN"
```

### Step 2: 收集诊断数据

```python
diagnostic_checklist = {
    # A. Loss曲线
    "training_loss": "持续下降？平滑？",
    "validation_loss": "上升？下降？拐点在哪？",

    # B. Eval metrics
    "eval_by_domain": {
        "general": "趋势？",
        "math": "趋势？",
        "code": "趋势？",
    },

    # C. 训练配置变化
    "data_mixture_changes": "何时改变？改变了什么？",
    "lr_schedule": "当前学习率？",
    "checkpoint_location": "问题出现在哪个step？",

    # D. 数据统计
    "new_datasets": "新增了哪些数据？",
    "sampling_epochs": "各数据源被采样多少轮？",

    # E. 对比实验
    "ablation_results": "移除新数据后如何？",
    "checkpoint_comparison": "不同阶段checkpoint对比？",
}
```

### Step 3: 决策树

```
Loss↓ + Eval↓ 问题
    │
    ├─ Validation loss 也上升？
    │   ├─ YES → 经典过拟合
    │   └─ NO → 继续诊断
    │
    ├─ 不同domain趋势不同？
    │   ├─ YES → 数据分布偏移
    │   └─ NO → 继续诊断
    │
    ├─ 旧任务性能下降 > 5%？
    │   ├─ YES → 灾难性遗忘
    │   └─ NO → 继续诊断
    │
    ├─ 最近改变了数据分布？
    │   ├─ YES → 数据分布问题
    │   └─ NO → 继续诊断
    │
    ├─ 学习率接近0？
    │   ├─ YES → LR调度问题
    │   └─ NO → 继续诊断
    │
    └─ 新增了数据源？
        ├─ YES → 数据质量问题
        └─ NO → 需要更深入调查
```

---

## 针对SmolLM3 Stage 3的分析

基于配置文件`stage3_9T_11T.yaml`：

### 可能的主要原因（按概率排序）

#### 1. 数据分布偏移 + 灾难性遗忘（最可能，~60%）

**证据**：
```yaml
# 激进的数据变化
General Web: 70% → 50% (-29%)
Math: 6.5% → 12.5% (+92%)
Code: 13% → 23% (+77%)
Reasoning: 0% → 2.2% (new)
```

**机制**：
- 通用数据大幅减少 → 通用能力更新频率降低
- Math/Code数据激增 → 梯度主要由这些数据主导
- 权重逐渐偏向math/code → 通用能力被覆盖

**预期表现**：
- ✓ Training loss ↓ (在math/code上优化)
- ✗ HellaSwag, ARC, MMLU ↓ (通用能力遗忘)
- ✓ GSM8K, MATH, HumanEval ↑ (math/code能力提升)

#### 2. LR Decay期间数据变化（次可能，~25%）

**证据**：
```yaml
lr_decay_starting_step: 4198001  # Stage 3开始
lr_decay_style: linear
min_decay_lr: 0  # 衰减到0！
```

**机制**：
- 低LR + 数据分布变化 = 难以平衡新旧知识
- 模型"僵化"在math/code特化状态
- 无法修正通用能力的下降

#### 3. 合成数据质量问题（可能，~10%）

**证据**：
```yaml
# 新增大量reasoning合成数据
- openmathinstruct-2
- openmathreasoning-4k
- open-codereasoning-4k
- natural_reasoning
- tiny-gsm-mind-*
- dolmino_math_synth_*
```

**风险**：
- 合成数据可能包含artifacts
- 过度拟合特定推理模式
- 降低真实场景泛化能力

#### 4. 过度训练（较小可能，~5%）

**证据**：
- 总训练11T tokens，已经非常多
- 在最后1T token继续训练可能超过最优点

---

## 通用解决方案框架

### 预防措施（Best Practices）

```yaml
# 1. 数据分布变化要渐进
✓ Good:
  Stage 2: General 70%, Math 10%, Code 20%
  Stage 2.5: General 65%, Math 15%, Code 20%
  Stage 3: General 60%, Math 20%, Code 20%

✗ Bad:
  Stage 2: General 70%, Math 10%, Code 20%
  Stage 3: General 50%, Math 25%, Code 25%  # 突变

# 2. 在Stable phase调整数据，不在Decay phase
✓ Good:
  Stage 2 (Stable): 调整数据分布
  Stage 3 (Stable): 稳定训练
  Stage 4 (Decay): 仅LR衰减，数据不变

✗ Bad:
  Stage 3 (Decay): 同时改数据和降LR

# 3. 保持足够的通用数据比例
✓ Good:
  General data: 始终 > 50%

✗ Bad:
  General data: 降到30-40%

# 4. LR不要衰减到0
✓ Good:
  min_decay_lr: 2e-5

✗ Bad:
  min_decay_lr: 0

# 5. 密集的evaluation监控
✓ Good:
  eval_interval: 2000 steps
  monitor: multiple domains

✗ Bad:
  eval_interval: 6000 steps
  monitor: only training loss
```

### 补救措施（Remediation）

```python
# 如果已经遇到问题

# 选项1: 回滚到问题前的checkpoint
best_checkpoint = "step_4198000"  # Stage 3开始前

# 选项2: Early stopping
# 找到eval开始下降的点
optimal_checkpoint = find_optimal_checkpoint(eval_log)

# 选项3: Model merging
final_model = (
    0.7 * checkpoint_before_divergence +
    0.3 * checkpoint_after_divergence
)

# 选项4: 继续训练但调整配置
new_config = {
    "data_mixture": "恢复更高的general比例",
    "learning_rate": "重新warmup",
    "eval_interval": "更频繁监控",
}

# 选项5: 针对性的continual training
# 用通用数据做短期训练，恢复通用能力
continual_training(
    data="general only",
    steps=50000,
    lr=1e-5,
)
```

---

## 总结

### 核心原则

1. **Training Loss不是唯一指标**
   - 必须同时监控eval metrics
   - 不同domain分别监控

2. **数据分布变化需谨慎**
   - 渐进式变化
   - 保持主要数据源的稳定性
   - 在stable phase调整，不在decay phase

3. **学习率调度要合理**
   - 不要衰减到0
   - 避免在低LR时改变数据

4. **Eval benchmark要匹配训练目标**
   - 明确优化目标
   - 接受合理的trade-off

5. **Early detection is key**
   - 频繁evaluation
   - 多domain监控
   - 设置alert机制

### 诊断优先级

遇到"Loss↓ Eval↓"时，按此顺序检查：

1. ✅ 检查validation loss（是否过拟合）
2. ✅ 检查不同domain表现（是否分布偏移）
3. ✅ 检查最近的配置变化（数据/LR）
4. ✅ 对比不同checkpoint（找到divergence点）
5. ✅ 分析新增数据（质量问题）
6. ✅ 计算forgetting rate（灾难性遗忘）

### 最后的建议

**对于LLM训练者**：
- 宁可保守也不要激进
- 数据和LR不要同时大改
- Eval比Loss重要
- 保留足够的checkpoint用于回滚

**对于SmolLM3 specific case**：
- 需要实际数据验证问题确实存在
- 如果存在，最可能是数据分布偏移+灾难性遗忘
- 建议使用Stage 2末期checkpoint或model merging

---

**文档创建**: 2025-11-12
**适用场景**: LLM Pretraining中的Loss-Eval Divergence问题
