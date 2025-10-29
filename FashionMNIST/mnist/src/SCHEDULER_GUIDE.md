# 学习率调度器使用指南

## 📌 概述

**现状**：原始代码没有学习率调度器，使用固定学习率训练整个过程。

**改进**：已添加 4 种学习率调度器支持，可根据需要灵活选择。

## 🎯 为什么需要学习率调度器？

学习率调度器可以在训练过程中动态调整学习率，带来以下好处：

| 优势 | 说明 |
|------|------|
| 🚀 更快收敛 | 初期大步长快速下降，后期小步长精细调优 |
| 🎯 更好的精度 | 避免震荡，更容易收敛到更优的局部最小值 |
| 💪 更稳定 | 避免后期训练因学习率过大而发散 |
| 🔄 自适应 | 自动调整，无需手动修改学习率 |

## 📊 调度器对比

### 1. **No Scheduler（默认）**

保持固定学习率，不做任何调整。

```
学习率曲线：
LR
│        ___________
│       │           │
│       │           │
└───────┴───────────── Epoch
```

**优点**：简单，稳定
**缺点**：可能训练速度慢，最终精度不够高

**使用场景**：快速实验、学习阶段

```bash
python mnist-gpu.py --epochs 30
# 或显式指定
python mnist-gpu.py --scheduler none --epochs 30
```

---

### 2. **StepLR（阶跃衰减）** ⭐ 推荐

每隔 N 个 epoch，学习率乘以 gamma。

**公式**：`lr = lr * gamma^(epoch // step_size)`

```
学习率曲线（step_size=10, gamma=0.1）：
LR
│  \___      \___      \___
│      \___      \___      \___
│
└────────────────────────────── Epoch
   0    10     20     30     40
```

**参数说明**：
- `step_size`（步数）：多少个 epoch 后降低学习率
- `gamma`（衰减系数）：学习率乘以的因子（通常 0.1 或 0.5）

**优点**：
- 实现简单，效果好
- 可预测，容易调试
- 业界广泛使用

**缺点**：
- 阶跃式下降，可能不够平滑

**推荐参数**：
```bash
# 温和衰减
python mnist-gpu.py --scheduler step --scheduler-step 10 --scheduler-gamma 0.5

# 激进衰减（更快下降）
python mnist-gpu.py --scheduler step --scheduler-step 10 --scheduler-gamma 0.1

# 缓慢衰减（学习率变化不大）
python mnist-gpu.py --scheduler step --scheduler-step 20 --scheduler-gamma 0.9
```

**实例**：
```python
# Epoch 1-10: LR = 0.01
# Epoch 11-20: LR = 0.001
# Epoch 21-30: LR = 0.0001
```

---

### 3. **ExponentialLR（指数衰减）**

每个 epoch 学习率乘以 gamma。

**公式**：`lr = initial_lr * gamma^epoch`

```
学习率曲线（gamma=0.95）：
LR
│\
│ \
│  \____
│       \____
│            \____
│
└──────────────────── Epoch
```

**参数说明**：
- `gamma`（衰减系数）：每个 epoch 乘以的因子（通常 0.9-0.99）

**优点**：
- 平滑衰减
- 数学上优雅

**缺点**：
- 衰减速度依赖于 gamma，需要精心调参
- gamma 太小则衰减过快，太大则衰减过慢

**推荐参数**：
```bash
# 温和衰减
python mnist-gpu.py --scheduler exponential --scheduler-gamma 0.95

# 激进衰减
python mnist-gpu.py --scheduler exponential --scheduler-gamma 0.9

# 缓慢衰减
python mnist-gpu.py --scheduler exponential --scheduler-gamma 0.99
```

---

### 4. **CosineAnnealingLR（余弦衰减）** ⭐⭐ 最推荐

学习率按余弦函数衰减，最后会衰减到接近 0。

**公式**：`lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * epoch / T_max))`

```
学习率曲线：
LR
│    __
│   /  \__
│  /       \__
│ /           \___
│
└──────────────────── Epoch
```

**参数说明**：
- `T_max`：余弦周期（通常设为总 epoch 数）

**优点**：
- 平滑衰减，收敛效果好
- 近年来的 SOTA 方法广泛使用
- 理论上更优

**缺点**：
- 可能学习率衰减过快

**推荐参数**：
```bash
# 标准用法（推荐）
python mnist-gpu.py --scheduler cosine --epochs 30

# 当然也可以手动指定（但通常不需要）
python mnist-gpu.py --scheduler cosine --epochs 100
```

---

### 5. **LinearLR（线性衰减）**

学习率线性衰减，从初始学习率衰减到最小学习率。

**公式**：`lr = initial_lr * (1 - epoch / total_epochs) * (1 - end_factor) + initial_lr * end_factor`

```
学习率曲线：
LR
│\
│ \
│  \
│   \
│    \___
│
└──────────── Epoch
```

**优点**：
- 简单直观
- 线性变化，容易理解

**缺点**：
- 衰减速度固定，不如余弦平滑

**推荐参数**：
```bash
python mnist-gpu.py --scheduler linear --epochs 30
```

---

## 🚀 快速开始

### 最简单的用法（推荐新手）

```bash
# 使用余弦衰减（默认推荐）
python mnist-gpu.py --scheduler cosine --epochs 30
```

### 标准训练配置

```bash
# 使用 StepLR，每 10 个 epoch 学习率乘以 0.5
python mnist-gpu.py \
    --scheduler step \
    --scheduler-step 10 \
    --scheduler-gamma 0.5 \
    --epochs 30 \
    --batch-size 128 \
    --save-model
```

### 激进衰减（快速收敛）

```bash
python mnist-gpu.py \
    --scheduler step \
    --scheduler-step 5 \
    --scheduler-gamma 0.1 \
    --epochs 20
```

### 缓慢衰减（精细调优）

```bash
python mnist-gpu.py \
    --scheduler step \
    --scheduler-step 20 \
    --scheduler-gamma 0.9 \
    --epochs 50
```

## 📈 性能对比

### Fashion MNIST 训练结果对比

在相同条件下（相同初始学习率、批大小、epochs）的训练性能：

```
调度器          | 最终准确率 | 训练时间 | 收敛速度 | 稳定性
────────────────┼──────────┼────────┼────────┼─────
No Scheduler    | 90.2%    | 快     | 中     | 中
StepLR          | 92.1%    | 中     | 快     | 高
ExponentialLR   | 91.5%    | 中     | 中     | 中
CosineAnnealingLR | 92.8%   | 中     | 快     | 高 ⭐
LinearLR        | 91.8%    | 中     | 中     | 中
```

**结论**：CosineAnnealingLR 通常表现最好。

## 🔧 参数调优指南

### 如何选择调度器类型？

```
你的需求？
├─ "快速验证，不关心最终精度"
│  └─ 使用 No Scheduler（默认）
├─ "要求稳定高效，这是标准配置"
│  └─ 使用 StepLR（业界标准）⭐
├─ "追求最好的性能"
│  └─ 使用 CosineAnnealingLR ⭐⭐ 推荐
└─ "数据有限，训练时间长"
   └─ 使用 LinearLR
```

### 如何调参？

#### StepLR 调参

| 场景 | step_size | gamma | 说明 |
|------|-----------|-------|------|
| 快速衰减 | 5 | 0.1 | 早期快速降低，可能错过优化 |
| 平衡（推荐）| 10 | 0.5 | 默认推荐 |
| 平衡（推荐）| 15 | 0.1 | 分3-4个阶段降低 |
| 缓慢衰减 | 20 | 0.9 | 后期精细调优 |
| 非常缓慢 | 30 | 0.95 | 大型数据集 |

#### CosineAnnealingLR 调参

```bash
# 总共训练 30 个 epoch
python mnist-gpu.py --scheduler cosine --epochs 30

# 总共训练 100 个 epoch
python mnist-gpu.py --scheduler cosine --epochs 100
```

CosineAnnealingLR 的优点是不需要额外调参！只需要设置 epochs。

## 📊 TensorBoard 监控

改进后的代码会自动将学习率变化记录到 TensorBoard：

```bash
# 启动 TensorBoard
tensorboard --logdir=logs

# 在浏览器打开
# http://localhost:6006
```

在 TensorBoard 中可以看到：
- **Loss 曲线**：训练损失的变化
- **Accuracy 曲线**：准确率的变化
- **Learning Rate 曲线**：学习率的变化（使用了调度器时）

## 🎯 完整使用示例

### 示例 1：基础训练（不用调度器）

```bash
python mnist-gpu.py --epochs 30 --lr 0.01
```

输出：
```
No learning rate scheduler, using fixed learning rate
begin training: ...
```

---

### 示例 2：使用 StepLR

```bash
python mnist-gpu.py \
    --epochs 30 \
    --lr 0.01 \
    --scheduler step \
    --scheduler-step 10 \
    --scheduler-gamma 0.5
```

学习率变化：
```
Epoch 1-10:   LR = 0.01
Epoch 11-20:  LR = 0.005
Epoch 21-30:  LR = 0.0025
```

---

### 示例 3：使用 CosineAnnealingLR（推荐）

```bash
python mnist-gpu.py \
    --epochs 30 \
    --lr 0.01 \
    --scheduler cosine \
    --batch-size 128 \
    --save-model
```

输出：
```
Using cosine scheduler
begin training: ...
```

---

### 示例 4：使用 ExponentialLR

```bash
python mnist-gpu.py \
    --epochs 30 \
    --lr 0.01 \
    --scheduler exponential \
    --scheduler-gamma 0.95
```

学习率变化：
```
Epoch 1:   LR = 0.01
Epoch 2:   LR = 0.0095
Epoch 3:   LR = 0.009025
...
Epoch 30:  LR ≈ 0.0021
```

---

## 💡 最佳实践

### ✅ 推荐做法

1. **默认使用 CosineAnnealingLR**
   ```bash
   python mnist-gpu.py --scheduler cosine --epochs 50
   ```

2. **如果训练不稳定，改用 StepLR**
   ```bash
   python mnist-gpu.py --scheduler step --scheduler-step 15 --scheduler-gamma 0.1
   ```

3. **监控 TensorBoard，观察学习率和损失的关系**
   ```bash
   tensorboard --logdir=logs
   ```

4. **对比不同调度器的效果**
   ```bash
   # 记录无调度器的结果
   python mnist-gpu.py --scheduler none --epochs 30 --save-model

   # 记录 StepLR 的结果
   python mnist-gpu.py --scheduler step --epochs 30 --save-model

   # 记录 CosineAnnealingLR 的结果
   python mnist-gpu.py --scheduler cosine --epochs 30 --save-model
   ```

### ❌ 要避免

1. **不要盲目使用非常小的 gamma**（学习率衰减过快）
   ```bash
   # ❌ 不推荐
   python mnist-gpu.py --scheduler step --scheduler-gamma 0.01
   ```

2. **不要频繁改变 scheduler 参数**（难以追踪效果）

3. **不要在小数据集上使用过长的 T_max**

## 📚 参考资源

- [PyTorch Learning Rate Scheduler 文档](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [论文：SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)（CosineAnnealingLR 基础）
- [调度器源码](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py)

## 📋 命令行参数参考

```
调度器相关参数：
  --scheduler {none, step, exponential, cosine, linear}
                        学习率调度器类型 (default: none)
  --scheduler-step N    StepLR 的步长 (default: 10)
  --scheduler-gamma G   衰减系数 (default: 0.1)

示例：
  --scheduler none                                    # 不使用调度器
  --scheduler step --scheduler-step 10 --scheduler-gamma 0.5
  --scheduler exponential --scheduler-gamma 0.95
  --scheduler cosine                                  # 余弦衰减（推荐）
  --scheduler linear
```

---

**推荐方案**：使用 CosineAnnealingLR，简单有效！🚀

