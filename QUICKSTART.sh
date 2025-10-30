#!/bin/bash

# 快速开始脚本 - 增量训练

echo "======================================"
echo "🚀 Fashion MNIST 增量训练快速开始"
echo "======================================"
echo ""

# 进入源代码目录
cd FashionMNIST/mnist/src

echo "📁 当前目录: $(pwd)"
echo ""

# 显示可用的模型
echo "📦 可用的预训练模型:"
ls -lh ../../../model/
echo ""

# 建议的命令
echo "💡 推荐命令："
echo ""
echo "【方案1】轻度微调（推荐首选）:"
echo "python finetune-gpu.py \\"
echo "  --model-path=../../../model/mnist_cnn3.pt \\"
echo "  --batch-size=256 \\"
echo "  --lr=0.001 \\"
echo "  --epochs=30 \\"
echo "  --scheduler=cosine"
echo ""
echo "【方案2】头部微调（冻结卷积层）:"
echo "python finetune-gpu.py \\"
echo "  --model-path=../../../model/mnist_cnn3.pt \\"
echo "  --batch-size=256 \\"
echo "  --lr=0.001 \\"
echo "  --epochs=30 \\"
echo "  --freeze-conv \\"
echo "  --scheduler=cosine"
echo ""
echo "【方案3】深度微调（更激进）:"
echo "python finetune-gpu.py \\"
echo "  --model-path=../../../model/mnist_cnn3.pt \\"
echo "  --batch-size=512 \\"
echo "  --lr=0.002 \\"
echo "  --epochs=50 \\"
echo "  --scheduler=cosine \\"
echo "  --min-lr=0.00005"
echo ""

# 测试模型
echo "【测试模型是否正常】:"
echo "python test-model.py --model-path=../../../model/mnist_cnn3.pt"
echo ""

echo "💬 详细说明请查看根目录的 FINETUNE_GUIDE.md"
echo ""
