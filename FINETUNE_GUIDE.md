# 增量训练指南 (Fine-tuning Guide)

## 快速开始

基于预训练模型进行增量训练，使用较小的学习率来微调已学到的特征：

```bash
cd FashionMNIST/mnist/src

# 基础用法（使用默认参数）
python finetune-gpu.py --model-path=../../model/mnist_cnn3.pt

# 或指定你的模型路径
python finetune-gpu.py --model-path=../../../model/mnist_cnn3.pt
```

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | **必需** | 预训练模型的路径 |
| `--batch-size` | 256 | 批次大小（256-512推荐） |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 0.001 | **学习率（微调用，很小！）** |
| `--min-lr` | 0.00001 | 最小学习率 |
| `--scheduler` | cosine | 学习率调度器 |

### 高级参数

| 参数 | 说明 |
|------|------|
| `--freeze-conv` | 冻结卷积层，只微调FC层（推荐用于数据少的情况） |
| `--freeze-bn` | 冻结BatchNorm层的参数更新 |
| `--dropout` | Dropout率（默认0.3） |

## 实用示例

### 方案1：轻微调（推荐，一般情况）
```bash
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=256 \
  --lr=0.001 \
  --epochs=50 \
  --scheduler=cosine
```
**适用场景**: 已有很好的预训练模型，只需小幅改进

### 方案2：深度微调（数据量充足）
```bash
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=512 \
  --lr=0.002 \
  --epochs=100 \
  --scheduler=cosine \
  --min-lr=0.00005
```
**适用场景**: 想要较大幅度的改进

### 方案3：头部微调（数据量不足）
```bash
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=128 \
  --lr=0.001 \
  --epochs=30 \
  --freeze-conv \
  --scheduler=cosine
```
**适用场景**: 数据量少，只微调分类头（FC层）

### 方案4：保守微调（避免破坏已学特征）
```bash
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=256 \
  --lr=0.0005 \
  --epochs=30 \
  --scheduler=cosine \
  --min-lr=0.00001 \
  --dropout=0.2
```
**适用场景**: 预训练模型很好，害怕过度训练

## 关键概念

### 1. 微调学习率为什么这么小？
- 初始训练：lr = 0.04（从零开始学习）
- 微调训练：lr = 0.001（保持已学知识，小幅调整）
- **规则**: 微调学习率 = 初始学习率 / 50 ~ 100

### 2. 冻结层的选择

**`--freeze-conv`**: 冻结卷积层
```python
# 卷积层保持预训练的权重
# 只训练FC层的权重
```
- ✅ 优点：避免破坏底层特征，训练快
- ❌ 缺点：改进空间有限
- 🎯 何时用：数据量少（<10k样本），或模型已经很好

**`--freeze-bn`**: 冻结BatchNorm
- 固定BN的统计信息（mean, variance）
- 常用于微调大型预训练模型

**都不冻结**：完全微调
- 所有层都参与训练
- ✅ 优点：改进空间大
- ❌ 缺点：容易过拟合，需要足够数据

### 3. 数据增强自动调整

微调脚本已**自动使用轻度数据增强**：
```python
RandomRotation(10),      # 只旋转±10度
RandomCrop(28, padding=2) # 轻度裁剪
```
（相比初始训练的激进增强）

## 监控训练

### 使用TensorBoard查看训练过程
```bash
tensorboard --logdir=logs_finetune
```

### 关键指标
- **Validation Loss**: 应该继续下降
- **Accuracy**: 应该逐步提高
- **Learning Rate**: 应该逐步衰减（如果使用cosine）

## 输出文件

训练完成后，模型保存在：
```
FashionMNIST/mnist/data/mnt/mnist_cnn_finetuned.pt
```

如果want custom目录：
```bash
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --save-model-dir=../../../result
```

## 常见问题

### Q1: 微调后性能反而下降了？
**A**:
- ✓ 降低学习率（从0.001改为0.0005）
- ✓ 减少epochs（从50改为30）
- ✓ 尝试冻结卷积层：`--freeze-conv`

### Q2: 需要训练多久？
**A**: 取决于目标精度：
- 微调现有模型：5-20 epoch通常就够了
- 初始训练：100-200 epoch

### Q3: 什么时候用`--freeze-conv`？
**A**:
- 数据量很少（<5k）
- 预训练模型已经很好（>90%）
- 想保证稳定性

### Q4: 能否继续在微调的模型上再微调？
**A**: 可以的！
```bash
# 第一轮微调
python finetune-gpu.py --model-path=../../../model/mnist_cnn3.pt

# 第二轮微调（基于第一轮结果）
python finetune-gpu.py --model-path=../../../result/mnist_cnn_finetuned.pt --lr=0.0005 --epochs=20
```

## 性能对标

基于V100显卡，预期的训练速度：
- Batch size=256: ~50ms/batch
- 完整epoch (60k样本): ~50秒
- 50 epochs: ~42分钟

## 完整工作流示例

```bash
# 1. 进入源代码目录
cd FashionMNIST/mnist/src

# 2. 第一阶段：轻度微调
python finetune-gpu.py \
  --model-path=../../../model/mnist_cnn3.pt \
  --batch-size=256 \
  --lr=0.001 \
  --epochs=30 \
  --scheduler=cosine

# 3. 查看结果（检查logs_finetune中的TensorBoard）

# 4. 如果效果好，继续第二阶段微调（更激进）
python finetune-gpu.py \
  --model-path=../../../result/mnist_cnn_finetuned.pt \
  --batch-size=512 \
  --lr=0.0005 \
  --epochs=20

# 5. 比较最终模型与初始模型
# 结果都在 FashionMNIST/mnist/data/mnt/ 下
```

## 注意事项

⚠️ **重要**:
- 微调的学习率必须比初始学习率小得多
- 否则可能"遗忘"预训练学到的知识（灾难性遗忘）
- 建议从0.001开始，根据效果调整到0.0005或0.002

✅ **最佳实践**:
1. 先用保守参数试一次 (lr=0.001, epochs=20)
2. 检查精度是否有改进
3. 如果有改进，再考虑更激进的参数
4. 总是保存最佳模型（脚本会自动做）
