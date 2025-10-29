# Fashion MNIST GPU支持指南

## 概述

改进后的 `mnist-gpu.py` 现在支持多种GPU后端：

- **CUDA**: NVIDIA GPU (推荐用于最佳性能)
- **MPS**: Mac GPU (Metal Performance Shaders，Apple Silicon M1/M2/M3+)
- **CPU**: 通用处理器 (备选)

## 系统要求

### CUDA支持
- NVIDIA GPU 显卡
- CUDA Toolkit 10.0+
- cuDNN 7.0+
- PyTorch with CUDA support

### Mac GPU支持 (MPS)
- Mac 电脑配备 Apple Silicon (M1, M2, M3 等)
- macOS 12.0+
- PyTorch >= 1.12.0
- Metal Performance Shaders 已内置于 macOS

### CPU支持
- 任何支持的操作系统
- PyTorch CPU 版本

## 使用方法

### 1. 自动GPU检测（推荐）

默认情况下，脚本会自动检测并使用可用的GPU：

```bash
# 在 Mac 上（使用 MPS 如果可用）
python mnist-gpu.py --epochs 10

# 在 NVIDIA GPU 上（使用 CUDA）
python mnist-gpu.py --epochs 10

# 在 CPU 上（如果没有GPU可用）
python mnist-gpu.py --epochs 10
```

### 2. 强制使用特定设备

```bash
# 强制使用 CPU（禁用所有GPU）
python mnist-gpu.py --no-cuda --epochs 10

# 在 Mac 上禁用 MPS，使用 CPU
python mnist-gpu.py --no-mps --epochs 10
```

## 常见命令示例

### Mac 上进行完整训练

```bash
# 使用默认参数（自动使用MPS）
python mnist-gpu.py \
    --epochs 30 \
    --batch-size 128 \
    --test-batch-size 500 \
    --lr 0.001 \
    --save-model \
    --save-model-dir ./models

# 使用 CPU（如果想避免 MPS）
python mnist-gpu.py \
    --no-mps \
    --epochs 30 \
    --batch-size 64
```

### NVIDIA GPU 上进行训练

```bash
# 使用 CUDA（默认如果可用）
python mnist-gpu.py \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.01 \
    --save-model

# 强制 CPU（禁用 CUDA）
python mnist-gpu.py \
    --no-cuda \
    --epochs 10
```

## 主要改进点

### 1. GPU 检测逻辑
```python
优先级顺序：
CUDA (if available)
  ↓
MPS (if available on Mac)
  ↓
CPU (fallback)
```

### 2. 数据加载优化

不同设备的数据加载参数配置：

| 设备 | num_workers | pin_memory | 说明 |
|------|------------|-----------|------|
| CUDA | 1 | True | 启用多进程和内存锁定 |
| MPS | 0 | False | 单进程（MPS不支持pin_memory） |
| CPU | 0 | False | 最少配置 |

### 3. 分布式训练警告

Mac MPS 目前不支持 PyTorch 的 `DistributedDataParallel`，如果检测到该模式会发出警告。

## 性能对比（参考数据）

在相同的 Fashion MNIST 训练任务上，预期性能（相对于 CPU）：

- NVIDIA CUDA: **50-100x** 更快
- Mac MPS (M1/M2): **8-15x** 更快
- CPU: **1x** (基准)

具体性能取决于：
- GPU 型号
- 批处理大小
- 数据集大小
- 模型复杂度

## 故障排除

### 问题：在 Mac 上显示"GPU not available"

**原因1**：PyTorch 版本过旧（需要 >= 1.12.0）

```bash
# 检查版本
python -c "import torch; print(torch.__version__)"

# 升级 PyTorch
pip install --upgrade torch
```

**原因2**：Mac 芯片不支持 MPS（需要 Apple Silicon）

```bash
# 检查芯片架构
uname -m
# 输出 arm64 表示 Apple Silicon，x86_64 表示 Intel
```

### 问题：MPS 出现内存错误

**解决方案**：

1. 减少批处理大小：
```bash
python mnist-gpu.py --batch-size 32 --test-batch-size 100
```

2. 强制使用 CPU：
```bash
python mnist-gpu.py --no-mps
```

### 问题：CUDA 显示"Out of Memory"

**解决方案**：

1. 减少批处理大小
2. 减少模型大小
3. 检查是否有其他进程占用 GPU 内存

```bash
# NVIDIA GPU 内存监控
nvidia-smi
```

## 代码架构

### GPU 检测部分
```python
# 位置：main() 函数中 ========== 2. 检查并设置GPU设备 ==========
if not args.no_cuda:
    if torch.cuda.is_available():
        use_cuda = True
    elif torch.backends.mps.is_available() and not args.no_mps:
        use_mps = True
```

### 设备设置部分
```python
# 位置：main() 函数中 ========== 5. 设置计算设备 ==========
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

### 数据加载配置
```python
# 位置：main() 函数中 ========== 7. 数据加载器配置 ==========
if use_cuda:
    kwargs = {'num_workers': 1, 'pin_memory': True}
elif use_mps:
    kwargs = {'num_workers': 0, 'pin_memory': False}
else:
    kwargs = {}
```

## 命令行参数参考

```
GPU 相关：
  --no-cuda          禁用所有 GPU（CUDA 和 MPS），使用 CPU
  --no-mps           禁用 Mac GPU（MPS），使用 CPU

训练参数：
  --epochs           训练轮数 (default: 30)
  --batch-size       训练批大小 (default: 100)
  --test-batch-size  测试批大小 (default: 500)
  --lr               学习率 (default: 0.001)
  --momentum         SGD 动量 (default: 0.5)
  --seed             随机种子 (default: 1)

其他：
  --save-model       保存模型
  --save-model-dir   模型保存目录 (default: /data/mnt)
  --dataset          数据集路径 (default: ../data)
  --dir              TensorBoard 日志目录 (default: logs)
  --log-interval     打印间隔 (default: 10)
```

## 在 Mac 上的最佳实践

1. **首次运行**：使用默认参数测试 MPS 是否工作正常
2. **性能调优**：如果出现内存问题，从较小的批大小开始
3. **监控训练**：使用 TensorBoard 监控训练进度
4. **模型保存**：始终使用 `--save-model` 保存最佳模型

```bash
# 推荐的 Mac 训练命令
python mnist-gpu.py \
    --epochs 30 \
    --batch-size 64 \
    --test-batch-size 200 \
    --lr 0.001 \
    --save-model \
    --save-model-dir ./models \
    --dir ./logs
```

## 更新日志

### v2.0 (当前)
- ✅ 添加 Mac MPS 支持
- ✅ 改进 GPU 检测逻辑
- ✅ 优化数据加载配置
- ✅ 添加详细的错误提示
- ✅ 兼容分布式训练标志

### v1.0 (原始)
- ✅ CUDA 支持
- ✅ CPU 备选
- ✅ TensorBoard 日志

## 相关链接

- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [PyTorch CUDA 文档](https://pytorch.org/docs/stable/cuda.html)
- [TensorBoard 使用指南](https://www.tensorflow.org/tensorboard)

## 许可证

遵循原始项目许可证
