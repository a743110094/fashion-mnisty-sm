# 学习率调度器 - 快速参考卡

## 🚀 最常用的 3 种用法

### 1️⃣ 不用调度器（默认）
```bash
python mnist-gpu.py --epochs 30
```
- 学习率固定
- 最简单
- 适合快速实验

---

### 2️⃣ 阶跃衰减 ⭐ 推荐
```bash
python mnist-gpu.py --scheduler step --scheduler-step 10 --scheduler-gamma 0.5 --epochs 30
```
- 每 10 个 epoch，学习率乘以 0.5
- 实用效果好
- 业界标准

---

### 3️⃣ 余弦衰减 ⭐⭐⭐ 最推荐
```bash
python mnist-gpu.py --scheduler cosine --epochs 30
```
- 平滑衰减
- 效果最好
- 无需额外调参

---

## 📊 对比表

| 方法 | 命令 | 优点 | 缺点 |
|------|------|------|------|
| **No Scheduler** | `--scheduler none` | 简单快速 | 精度低 |
| **StepLR** | `--scheduler step --scheduler-step 10 --scheduler-gamma 0.5` | 效果好，稳定 | 参数调优 |
| **CosineAnnealingLR** | `--scheduler cosine` | 效果最好，无需调参 | 衰减快 |
| **ExponentialLR** | `--scheduler exponential --scheduler-gamma 0.95` | 平滑 | 需调参 |
| **LinearLR** | `--scheduler linear` | 简单直观 | 效果一般 |

---

## 🎯 应该选哪个？

```
你的优先级是？
├─ 快速验证想法
│  └─ 不用调度器（--scheduler none）
│
├─ 需要稳定的生产环境
│  └─ StepLR（--scheduler step）⭐
│
├─ 追求最高精度
│  └─ CosineAnnealingLR（--scheduler cosine）⭐⭐⭐
│
└─ 喜欢尝试不同方法
   └─ 都试一遍！
```

---

## 📈 学习率衰减示意

### No Scheduler
```
0.01 ━━━━━━━━━━━━━━━━━━━━ (固定不变)
```

### StepLR (step=10, gamma=0.5)
```
0.01 ━━━━━━━━━ ↘
0.005        ━━━━━━━━━ ↘
0.0025               ━━━━━━━━━
```

### CosineAnnealingLR
```
0.01 ╱╲
0.007╱  ╲
     ╱    ╲___
```

---

## 💻 完整命令示例

### 例 1：快速测试（1 epoch，无调度器）
```bash
python mnist-gpu.py --epochs 1
```

### 例 2：标准训练（推荐，30 epochs）
```bash
python mnist-gpu.py \
  --epochs 30 \
  --batch-size 128 \
  --scheduler cosine \
  --save-model
```

### 例 3：长训练（100 epochs，更好的精度）
```bash
python mnist-gpu.py \
  --epochs 100 \
  --batch-size 64 \
  --scheduler step \
  --scheduler-step 20 \
  --scheduler-gamma 0.5 \
  --save-model
```

### 例 4：快速衰减（快速收敛）
```bash
python mnist-gpu.py \
  --epochs 30 \
  --scheduler step \
  --scheduler-step 5 \
  --scheduler-gamma 0.1
```

### 例 5：缓慢衰减（精细调优）
```bash
python mnist-gpu.py \
  --epochs 50 \
  --scheduler exponential \
  --scheduler-gamma 0.98
```

---

## 🔍 监控学习率变化

### 查看 TensorBoard
```bash
tensorboard --logdir=logs
```

在 TensorBoard 中可以看到：
- **Learning Rate 曲线**：学习率如何变化
- **Loss 曲线**：损失如何下降
- **Accuracy 曲线**：准确率如何上升

---

## ⚡ 参数速查表

### StepLR 参数

| 场景 | step | gamma | 说明 |
|------|------|-------|------|
| 快速 | 5 | 0.1 | 衰减快，可能欠优化 |
| 平衡 | 10 | 0.5 | 推荐使用 |
| 平衡 | 15 | 0.1 | 分多个阶段 |
| 缓慢 | 20 | 0.9 | 精细调优 |

### ExponentialLR 参数

| 场景 | gamma |
|------|-------|
| 快速 | 0.90 |
| 中等 | 0.95 |
| 缓慢 | 0.99 |

---

## ❓ 常见问题

**Q: CosineAnnealingLR 需要调参吗？**
A: 不需要！只需设置 `--epochs`

**Q: 哪个调度器效果最好？**
A: CosineAnnealingLR 通常最好

**Q: 能同时使用多个调度器吗？**
A: 当前不支持，但可以用 `ChainedScheduler`（高级用法）

**Q: 学习率会变成 0 吗？**
A: CosineAnnealingLR 会衰减到接近 0，这是正常的

**Q: 如何保存/加载调度器状态？**
A: `scheduler.state_dict()` 和 `scheduler.load_state_dict()`

---

## 📝 我的推荐顺序

### 1️⃣ 第一次尝试（推荐）
```bash
python mnist-gpu.py --scheduler cosine --epochs 30 --save-model
```
✅ 简单、有效、无需调参

### 2️⃣ 如果精度不够
```bash
python mnist-gpu.py --scheduler step --scheduler-step 15 --scheduler-gamma 0.1 --epochs 50
```
✅ 更长训练时间、更多阶段

### 3️⃣ 如果还是不够
```bash
python mnist-gpu.py --scheduler exponential --scheduler-gamma 0.97 --epochs 100
```
✅ 超长训练、平滑衰减

---

## 🎓 学习资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [深度学习最佳实践](https://cs231n.github.io/neural-networks-3/#annealing-the-learning-rate)

---

**最后建议**：默认使用 **CosineAnnealingLR**，99% 的情况下都够用！🚀
