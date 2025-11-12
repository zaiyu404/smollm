# SmolLM3 训练Loss与评测分歧问题分析报告

## 问题描述
在SmolLM3训练1T token后（约9T tokens处），观察到training loss持续下降，但评测指标（evaluation metrics）却出现下降，呈现loss与eval性能的分歧（divergence）。

## 根本原因分析

通过分析训练配置文件，发现问题出现在 **Stage 3 (Decay Stage)** 开始时。

### 训练阶段划分

| 阶段 | Token范围 | 训练步骤 | 关键特征 |
|------|-----------|----------|----------|
| Stage 1 | 0-8T | 1 - 3,450,000 | 初始数据混合，恒定学习率 |
| Stage 2 | 8T-9T | 3,450,001 - 4,198,000 | 调整数据混合，添加部分新数据源 |
| **Stage 3** | **9T-11T** | **4,198,001 - 4,720,000** | **Decay阶段：问题发生点** |

### Stage 3 的关键变化（问题源头）

#### 1. 学习率调度变化
```yaml
# 从配置文件 stage3_9T_11T.yaml (line 539-546)
learning_rate_scheduler:
  learning_rate: 0.0002
  lr_decay_starting_step: 4198001  # ← Stage 3开始
  lr_decay_steps: 522000           # 衰减52.2万步
  lr_decay_style: linear           # 线性衰减到0
  min_decay_lr: 0
```

**影响**: 学习率从 0.0002 线性衰减到 0

#### 2. 数据分布重大变化

##### A. 通用网页数据权重大幅降低
| 数据源 | Stage 1-2权重 | Stage 3权重 | 变化 |
|--------|--------------|------------|------|
| FineWeb-Edu | 0.333 (33.3%) | 0.20 (20%) | **-40%** |
| DCLM | 0.37 (37%) | 0.30 (30%) | **-19%** |
| PES2O | 0.02 (2%) | 0.002 (0.2%) | **-90%** |
| Wiki | 0.001 (0.1%) | 0.0002 (0.02%) | **-80%** |
| StackExchange | 0.004 (0.4%) | 0.001 (0.1%) | **-75%** |

##### B. 数学和代码数据权重激增
| 数据源 | Stage 1-2权重 | Stage 3权重 | 变化 |
|--------|--------------|------------|------|
| Python代码 | 0.025 (2.5%) | 0.07 (7%) | **+180%** |
| C++代码 | 0.018 (1.8%) | 0.044 (4.4%) | **+144%** |
| MegaMath-Text-Code | 0.02 (2%) | 0.05 (5%) | **+150%** |
| finemath-4plus | 0.02 (2%) | 0.025 (2.5%) | **+25%** |
| infiwebmath-4plus | 0.01 (1%) | 0.02 (2%) | **+100%** |
| SQL代码 | 0.006 (0.6%) | 0.013 (1.3%) | **+117%** |
| Jupyter Notebooks | 0.0055 (0.55%) | 0.012 (1.2%) | **+118%** |

##### C. 新增大量推理和合成数据（Stage 3独有）
```yaml
# 这些数据在Stage 1-2中不存在，仅在Stage 3添加：
- Cosmopedia2: 0.004 (0.4%)
- Multilingual Wiki: 0.008 (0.8%)
- OpenMathInstruct-2: 0.005 (0.5%)
- OpenMathReasoning-4k: 0.005 (0.5%)
- OpenCodeReasoning-4k: 0.0005 (0.05%)
- Natural Reasoning: 0.001 (0.1%)
- Tiny-GSM-Mind Problem Solving: 0.003 (0.3%)
- Tiny-GSM-Mind 2students: 0.003 (0.3%)
- Dolmino Math Synth GSM8K: 0.0004 (0.04%)
- Dolmino Math Synth Basic: 0.0002 (0.02%)
```

**总计新增推理数据**: 约 2.2% 的训练权重

#### 3. 数据分布对比总结

| 数据类型 | Stage 1-2总权重 | Stage 3总权重 | 趋势 |
|----------|----------------|--------------|------|
| 通用网页 (FineWeb-Edu + DCLM) | ~70% | ~50% | ↓↓↓ 大幅下降 |
| 代码相关 | ~13% | ~23% | ↑↑ 显著上升 |
| 数学相关 | ~6.5% | ~12.5% | ↑↑ 显著上升 |
| 推理/合成数据 | 0% | ~2.2% | ↑ 新增 |

## 为什么会导致Loss下降但Eval变差？

### 1. **数据分布偏移 (Distribution Shift)**
- **训练loss下降**: 模型在新的数据分布（大量math/code/reasoning数据）上继续学习，training loss自然下降
- **Eval性能下降**: 但评测benchmark主要是通用任务（HellaSwag, ARC, MMLU等），这些任务的数据分布与Stage 3的训练数据相差较大

### 2. **灾难性遗忘 (Catastrophic Forgetting)**
在学习率衰减阶段，模型对新数据（math/code/reasoning）的过度适应导致：
- 忘记了在通用数据上学到的知识
- 特别是FineWeb-Edu和DCLM的权重从70%降至50%，而这些是通用知识的主要来源

### 3. **过拟合特定领域 (Domain Overfitting)**
- 代码数据从13%→23%（+77%增长）
- 数学数据从6.5%→12.5%（+92%增长）
- 这种偏向可能使模型在代码和数学任务上表现提升，但牺牲了通用任务性能

### 4. **合成数据的副作用**
新增的2.2%推理数据（OpenMathInstruct, Dolmino等）是合成数据：
- 这些数据可能包含特定的模式或偏差
- 在学习率衰减阶段，模型可能过度记忆这些模式，而非真正理解

### 5. **学习率衰减期的脆弱性**
在LR decay阶段改变数据分布是**高风险操作**：
- 模型的可塑性降低（学习率从0.0002→0）
- 此时引入大量新数据源和权重变化，模型难以平衡新旧知识
- 容易导致对新数据的过拟合，而无法保持在原有分布上的泛化能力

## 受影响的Evaluation Benchmarks

根据 `text/evaluation/smollm3/smollm3_base.txt`，主要评测任务包括：

### 通用英语任务（最可能受影响）
- HellaSwag (常识推理)
- ARC (科学推理)
- MMLU / MMLU-Pro (多任务知识)
- BoolQ, CommonsenseQA, Winogrande (常识)
- OpenBookQA, PIQA (物理/常识推理)

### 数学任务（可能有所改善）
- GSM8K
- MATH

### 多语言任务（可能受影响）
- Belebele (阿拉伯语、俄语、中文、德语、法语等)
- Global MMLU (多语言)
- MLMM HellaSwag (多语言)
- Flores200 (翻译任务)

**预期结果**:
- 通用推理任务性能下降 ⚠️
- 数学/代码任务性能可能提升 ✓
- 多语言任务可能下降（多语言数据权重变化不大，但被math/code挤占）⚠️

## 解决方案建议

### 1. 立即诊断措施
```bash
# 检查不同checkpoint的评测结果对比
- 评测 step 4,198,000 的checkpoint（Stage 3开始前）
- 评测 step 4,400,000 的checkpoint（Stage 3中期）
- 评测 step 4,720,000 的checkpoint（Stage 3结束）
- 对比各个benchmark的性能曲线
```

### 2. 短期修复方案

#### 方案A: 回滚到Stage 2结束
- 使用 step 4,198,000 的checkpoint作为最终模型
- 跳过Stage 3的decay阶段
- **优点**: 保持通用性能
- **缺点**: 损失Stage 3学到的math/code能力

#### 方案B: 使用Stage 2末期进行温和的Decay
修改配置，仅做LR decay，不改变数据分布：
```yaml
# 新的decay配置建议
data_stages:
- data:
    # 保持Stage 2的数据分布不变
    dataset_weights:
      - 0.30   # FineWeb-Edu (保持)
      - 0.33   # DCLM (保持)
      # ... 其他权重保持Stage 2配置
  name: gentle decay
  start_training_step: 4198001

optimizer:
  learning_rate_scheduler:
    lr_decay_starting_step: 4198001
    lr_decay_steps: 522000
    min_decay_lr: 2e-5  # 不要衰减到0，保持小的学习率
```

### 3. 长期改进方案

#### 方案C: 重新设计训练计划
```yaml
训练阶段重新规划：

Stage 1 (0-7T): 通用预训练
  - FineWeb-Edu + DCLM为主 (70%)
  - 适量code (10%) 和 math (5%)

Stage 2 (7T-9T): 能力增强
  - 逐步增加math/code数据
  - FineWeb-Edu + DCLM: 70% → 60%
  - Math/Code: 15% → 25%

Stage 3 (9T-10.5T): 多能力平衡
  - 保持平衡的数据分布
  - FineWeb-Edu + DCLM: 55%
  - Math/Code: 30%
  - Reasoning: 5%
  - 恒定学习率

Stage 4 (10.5T-11T): 温和衰减
  - 保持Stage 3的数据分布不变
  - 仅做学习率衰减: 2e-4 → 2e-5
  - 较短的decay阶段（避免过度拟合）
```

#### 方案D: 添加正则化措施
```yaml
在Stage 3添加：
1. 更频繁的evaluation (每2000步而非6000步)
   eval_interval: 2000  # 当前是6000

2. Early stopping based on eval metrics
   - 监控通用benchmark (HellaSwag, ARC, MMLU)
   - 如果下降超过阈值，停止训练

3. 权重平均 (Weight Averaging)
   - 对Stage 3的多个checkpoint做EMA
   - 减少overfitting的影响
```

### 4. 数据策略改进

#### 建议的数据权重调整（Stage 3修改版）
```yaml
# 更保守的权重分配
dataset_weights:
  - 0.25  # FineWeb-Edu (当前0.20 → 建议0.25)
  - 0.32  # DCLM (当前0.30 → 建议0.32)
  - 0.01  # PES2O (当前0.002 → 建议0.01)

  # Math/Code适度增加，但不要过激
  - 0.04  # Python (当前0.07 → 建议0.04)
  - 0.025 # C++ (当前0.044 → 建议0.025)
  - 0.03  # MegaMath-Text-Code (当前0.05 → 建议0.03)

  # 推理数据减半
  - 0.0025 # OpenMathInstruct (当前0.005 → 建议0.0025)
  - 0.0025 # OpenMathReasoning (当前0.005 → 建议0.0025)
```

**目标**: 保持通用能力的同时，温和提升math/code性能

## 实验验证计划

### Experiment 1: Checkpoint对比
```bash
# 使用以下checkpoint评测所有benchmark
checkpoints=(
  "step-4198000"  # Stage 3前
  "step-4250000"
  "step-4300000"
  "step-4350000"
  "step-4400000"
  "step-4450000"
  "step-4500000"
  "step-4550000"
  "step-4600000"
  "step-4650000"
  "step-4700000"
  "step-4720000"  # Stage 3后
)

# 评测所有benchmark并绘制性能曲线
for ckpt in "${checkpoints[@]}"; do
  lighteval vllm "model_name=s3://smollm3/.../checkpoint-$ckpt" \
    "smollm3_base.txt" --custom-tasks "tasks.py"
done
```

### Experiment 2: 数据消融实验
重新训练Stage 3，测试不同数据配置：
1. Config A: 保持Stage 2数据分布
2. Config B: 仅增加20% math/code（而非当前的80-180%）
3. Config C: 不添加任何推理合成数据
4. Config D: 推荐的平衡配置（见上文）

### Experiment 3: 学习率消融
1. LR_A: 衰减到2e-5（而非0）
2. LR_B: 衰减到1e-5
3. LR_C: 保持2e-4恒定（不decay）
4. LR_D: Cosine衰减而非Linear

## 监控指标

在未来训练中，应该持续监控：

```python
# 关键指标
metrics_to_track = {
    # Loss指标
    "train_loss": "训练loss（应持续下降）",
    "eval_loss_general": "通用数据的validation loss",
    "eval_loss_math": "数学数据的validation loss",
    "eval_loss_code": "代码数据的validation loss",

    # Benchmark指标（每2000步评测）
    "hellaswag_acc": "常识推理（预期不应下降）",
    "arc_acc": "科学推理（预期不应下降）",
    "mmlu_acc": "多任务知识（预期不应下降）",
    "gsm8k_acc": "数学推理（可以提升）",
    "math_acc": "高级数学（可以提升）",

    # 分歧指标（divergence metrics）
    "train_eval_gap": "训练loss vs eval loss的差距（不应过大）",
    "general_vs_math_gap": "通用任务 vs 数学任务性能比（应保持平衡）",
}
```

## 参考文献与类似案例

类似问题在大模型训练中并不罕见：

1. **Llama 3**: 也使用了多阶段训练，但在最后阶段保持了数据分布的稳定性
2. **OLMo 2**: 强调了embedding layer的weight decay removal对稳定性的重要性（已在SmolLM3中采用）
3. **Chinchilla**: 研究表明数据分布变化应该是渐进式的，而非突变

## 总结

**问题的核心**: 在学习率衰减阶段（Stage 3, 9T-11T tokens）同时进行了激进的数据分布变化：
- 通用数据权重-40%
- Math/Code数据权重+80-180%
- 新增2.2%的推理合成数据

这导致：
- ✓ Training loss继续下降（在新数据分布上学习）
- ✗ Eval性能下降（通用benchmark上的泛化能力降低）

**建议优先级**:
1. 🔴 **立即**: 评测step 4,198,000 checkpoint，确认问题发生点
2. 🟠 **短期**: 考虑使用Stage 2末期checkpoint或重新训练Stage 3（保守的数据配置）
3. 🟡 **长期**: 重新设计训练计划，采用更温和的数据分布过渡策略

---

**分析日期**: 2025-11-12
**配置文件位置**:
- `/home/user/smollm/text/pretraining/smollm3/stage1_8T.yaml`
- `/home/user/smollm/text/pretraining/smollm3/stage2_8T_9T.yaml`
- `/home/user/smollm/text/pretraining/smollm3/stage3_9T_11T.yaml`
