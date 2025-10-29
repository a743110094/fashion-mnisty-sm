# Mac GPU 快速开始指南

## ⚡ 30 秒快速启动

```bash
# 在 Mac 上立即运行（自动使用 MPS）
python mnist-gpu.py --epochs 10
```

## ✅ 前置检查

### 检查 PyTorch 版本
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# 需要 >= 1.12.0
```

### 检查 MPS 可用性
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
# 应该输出: MPS available: True
```

### 检查芯片架构
```bash
uname -m
# arm64 = Apple Silicon (M1/M2/M3) ✓
# x86_64 = Intel (不支持 MPS) ✗
```

## 🚀 常用命令

### 基础训练
```bash
# 使用默认参数（推荐新手）
python mnist-gpu.py

# 快速测试（1 epoch）
python mnist-gpu.py --epochs 1

# 完整训练
python mnist-gpu.py --epochs 30 --save-model
```

### 性能优化
```bash
# 如果出现内存错误，减少批大小
python mnist-gpu.py --batch-size 32

# 快速训练（大批大小）
python mnist-gpu.py --batch-size 256 --test-batch-size 1000

# 最小化内存占用
python mnist-gpu.py --batch-size 16 --test-batch-size 64
```

### 故障排除
```bash
# 如果 MPS 有问题，强制使用 CPU
python mnist-gpu.py --no-mps

# 禁用所有 GPU
python mnist-gpu.py --no-cuda

# 查看详细的启动信息
python mnist-gpu.py --epochs 1 2>&1 | grep -E "(Using|available)"
```

### 监控训练
```bash
# 查看 TensorBoard
tensorboard --logdir=logs

# 然后在浏览器打开: http://localhost:6006
```

## 📊 预期性能

**Mac MPS 相对于 CPU 的加速倍数：**

| Mac 型号 | MPS vs CPU | 备注 |
|---------|-----------|------|
| M1 | 8-12x | 入门级 |
| M1 Pro | 12-18x | 专业版 |
| M2 | 10-15x | 改进版 |
| M2 Ultra | 20-25x | 最高性能 |

**Fashion MNIST 训练时间预估：**

- **MPS**: ~30秒/epoch (100张图片)
- **CPU**: ~3-5分钟/epoch
- **NVIDIA GPU**: ~1-2秒/epoch (参考)

## 🔧 调试技巧

### 查看设备信息
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Tensor device: {torch.zeros(1).device}")
```

### 运行时检查
脚本启动时会自动打印：
```
Using Mac GPU (MPS)
# 或
GPU not available, using CPU
```

### 常见问题排查

**问题**：显示 "GPU not available, using CPU"
```bash
# 解决方案 1: 升级 PyTorch
pip install --upgrade torch

# 解决方案 2: 检查您的 Mac 是否为 Apple Silicon
uname -m  # 应该输出 arm64
```

**问题**：MPS 训练很慢
```bash
# 这可能是模型太小，尝试更大的批大小
python mnist-gpu.py --batch-size 256
```

**问题**：内存溢出 (Out of Memory)
```bash
# 减少批大小
python mnist-gpu.py --batch-size 16
```

## 📝 完整训练脚本示例

### 基础训练 (推荐)
```bash
python mnist-gpu.py \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.001 \
    --save-model \
    --dir ./logs
```

### 快速验证
```bash
python mnist-gpu.py \
    --epochs 2 \
    --batch-size 128 \
    --log-interval 5
```

### 最大性能
```bash
python mnist-gpu.py \
    --epochs 50 \
    --batch-size 256 \
    --test-batch-size 1000 \
    --lr 0.01 \
    --save-model \
    --save-model-dir ./models
```

## 📌 重要提示

1. **第一次运行** - 需要下载 Fashion MNIST 数据集 (~30MB)
2. **MPS 限制** - 目前不支持分布式训练 (`--backend` 标志会被忽略)
3. **模型保存** - 使用 `--save-model` 保存训练结果
4. **日志查看** - 使用 TensorBoard 可视化训练过程

## 🔗 相关文档

- [README_GPU_SUPPORT.md](./README_GPU_SUPPORT.md) - 详细文档
- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [TensorBoard 使用](https://www.tensorflow.org/tensorboard)

## ⚙️ 硬件要求检查清单

- [ ] Mac with Apple Silicon (M1/M2/M3/M4)
- [ ] macOS 12.0 或更新版本
- [ ] PyTorch >= 1.12.0
- [ ] Python 3.7+
- [ ] 至少 2GB 可用 RAM

## 💡 性能提示

```bash
# 最佳平衡（大多数 Mac）
python mnist-gpu.py \
    --batch-size 128 \
    --epochs 30

# 内存受限的 Mac
python mnist-gpu.py \
    --batch-size 32 \
    --no-mps  # 如果出现问题

# 高性能 Mac (M1 Pro/Max)
python mnist-gpu.py \
    --batch-size 512 \
    --test-batch-size 2000
```

---

**需要帮助?** 查看 [README_GPU_SUPPORT.md](./README_GPU_SUPPORT.md) 了解更多详情。
