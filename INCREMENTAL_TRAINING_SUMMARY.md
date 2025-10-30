# 增量训练完整方案总结

## 📋 已创建的文件

### 1. **finetune-gpu.py** - 主训练脚本
位置: `FashionMNIST/mnist/src/finetune-gpu.py`

**功能**:
- ✅ 加载预训练模型
- ✅ 支持冻结部分层（卷积层/BatchNorm）
- ✅ 使用较小学习率微调
- ✅ 自动保存最佳模型
- ✅ 支持多种学习率调度器
- ✅ 集成TensorBoard可视化
- ✅ 早停机制

**关键特性**:
```python
# 微调学习率（相比初始训练的0.04，微调用0.001）
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 支持冻结卷积层
--freeze-conv  # 只训练FC层

# 轻度数据增强（避免过度增强破坏预训练特征）
RandomRotation(10)  # 只±10度
```

### 2. **test-model.py** - 模型测试脚本
位置: `FashionMNIST/mnist/src/test-model.py`

**功能**:
- ✅ 验证模型文件完整性
- ✅ 检查GPU可用性
- ✅ 测试前向传播
- ✅ 快速评估（3个batch）
- ✅ 显示内存使用情况

### 3. **FINETUNE_GUIDE.md** - 详细使用指南
位置: 根目录 `FINETUNE_GUIDE.md`

包含：
- 快速开始命令
- 参数详细说明
- 4种实用方案
- 常见问题解答
- 工作流示例

### 4. **QUICKSTART.sh** - 快速参考脚本
位置: 根目录 `QUICKSTART.sh`

快速查看推荐命令。

---

## 🎯 三步快速开始

### 第一步：测试模型
```bash
cd FashionMNIST/mnist/src
python test-model.py --model-path=../../../model/mnist_cnn3.pt
```

**预期输出**:
```
✅ 权重加载成功
✅ 前向传播成功
✅ 所有测试通过！模型可以进行微调训练
```

### 第二步：执行微调训练
```bash
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=256 \
  --lr=0.001 \
  --epochs=30 \
  --scheduler=cosine
```

### 第三步：查看结果
```bash
tensorboard --logdir=logs_finetune
# 打开浏览器访问 http://localhost:6006
```

---

## 📊 四种微调方案对比

| 方案 | 学习率 | Batch | 冻结卷积 | 适用场景 | 预计改进 |
|------|--------|-------|--------|---------|---------|
| **轻微调** | 0.001 | 256 | ❌ | 一般情况（推荐） | 0.5-1% |
| **头部微调** | 0.001 | 256 | ✅ | 数据少、模型好 | 0.2-0.5% |
| **深度微调** | 0.002 | 512 | ❌ | 数据充足、需大幅改进 | 1-2% |
| **保守微调** | 0.0005 | 256 | ❌ | 害怕过度训练 | 0.2-0.8% |

---

## 🔑 核心参数说明

### 学习率设置
```
初始训练学习率: 0.04
微调学习率推荐: 0.001 - 0.002

规则: 微调LR = 初始LR / 20-50
```

**为什么要用小学习率？**
- 预训练模型已经学到很多有用特征
- 大的学习率会"遗忘"这些特征（灾难性遗忘）
- 小学习率只做轻微调整

### Batch Size选择
```
V100 (32GB) 推荐:
- 轻微调: 256-512
- 深度微调: 512-1024
- 头部微调: 128-256
```

### Epochs设置
```
轻微调: 20-50 epoch
深度微调: 50-100 epoch
停止条件: 早停（validation loss无改进15次）
```

---

## 🎓 工作流示例

### 场景1：从91%提升到92-93%

```bash
# 第一次微调（保守）
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=256 \
  --lr=0.001 \
  --epochs=30 \
  --scheduler=cosine

# 观察结果，如果有改进，继续第二次微调
python finetune-gpu.py \
  --model-path=../../../result/mnist_cnn_finetuned.pt \
  --batch-size=256 \
  --lr=0.0005 \
  --epochs=20 \
  --scheduler=cosine
```

### 场景2：只优化分类层（已有很好特征）

```bash
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=256 \
  --lr=0.001 \
  --epochs=50 \
  --freeze-conv \
  --scheduler=cosine
```

---

## ⚙️ 配置建议

### 对于V100 (32GB)的优化
```python
# 数据加载器优化
num_workers: 8          # 不要用64（太多！）
pin_memory: True
batch_size: 256-512

# 显存足够，可用更大batch加快训练
```

### 监控指标
在TensorBoard中查看：
- **Accuracy**: 应该逐步提高
- **Validation Loss**: 应该继续下降
- **Learning Rate**: 应该逐步衰减

---

## 📈 预期性能提升

基于当前模型（91% acc）的现实预期：

```
轻微调 (lr=0.001, 30 epoch):  91.0% → 91.5-92.0%
头部微调 (freeze-conv):        91.0% → 91.2-91.5%
深度微调 (lr=0.002, 100 epoch): 91.0% → 92.0-93.0%
多轮微调:                      91.0% → 93.0-94.0% (需要多次实验)
```

**注**:
- 从91%→95%非常困难，可能需要架构改进（如ResNet）
- 微调通常只能改进1-2%
- 每个额外的百分点改进都变得更难

---

## ❌ 常见错误及解决

### 错误1: 模型精度反而下降
```
原因: 学习率太大或训练时间太长
解决:
✓ 降低学习率 (0.001 → 0.0005)
✓ 减少epochs (50 → 30)
✓ 启用 --freeze-conv
✓ 检查early_stopping是否正常工作
```

### 错误2: 训练很慢
```
原因: num_workers过多导致CPU瓶颈
解决:
✓ 减少num_workers (64 → 8)
✓ 增加batch_size (256 → 512)
✓ 检查数据增强是否过复杂
```

### 错误3: 显存溢出
```
解决:
✓ 减少batch_size (512 → 256)
✓ 减少num_workers
✓ 使用 --freeze-conv 减少计算量
```

---

## 🚀 下一步建议

1. **立即尝试**:
   - [ ] 运行test-model.py验证模型
   - [ ] 执行轻微调（方案1）
   - [ ] 观察TensorBoard结果

2. **如果需要更大改进**:
   - [ ] 尝试深度微调（方案3）
   - [ ] 调整数据增强策略
   - [ ] 考虑集成多个微调模型

3. **如果目标是95%+**:
   - [ ] 考虑使用更深网络（ResNet）
   - [ ] 或使用预训练的大模型（ImageNet预训练）
   - [ ] 这不是简单微调能达到的

---

## 📚 更多资源

- **PyTorch官方微调指南**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **TensorBoard使用**: `tensorboard --logdir=logs_finetune`
- **详细参数说明**: 见 `FINETUNE_GUIDE.md`

---

## 🎉 总结

✅ **已提供**:
- 完整的微调脚本 (finetune-gpu.py)
- 模型测试工具 (test-model.py)
- 详细使用指南 (FINETUNE_GUIDE.md)
- 4种实用方案

✅ **可以做**:
- 轻松在预训练模型基础上继续训练
- 冻结部分层加速训练
- 使用TensorBoard监控进度
- 自动保存最佳模型

🎯 **下一步**:
按照三步快速开始的指引执行微调训练！
