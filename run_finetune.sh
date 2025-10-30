#!/bin/bash

# 增量训练执行脚本
# 使用方法: bash run_finetune.sh [方案号]
# 例如: bash run_finetune.sh 1

set -e  # 出错时停止

cd "$(dirname "$0")"
cd FashionMNIST/mnist/src

PLAN=${1:-1}  # 默认方案1

echo "======================================"
echo "🚀 Fashion MNIST 增量训练"
echo "======================================"
echo ""

case $PLAN in
    1)
        echo "📋 方案1: 轻度微调（推荐首选）"
        echo "学习率: 0.001"
        echo "Batch Size: 256"
        echo "Epochs: 30"
        echo "Scheduler: Cosine"
        echo ""
        python finetune-gpu.py \
            --model-path=../../../model/mnist_cnn3.pt \
            --batch-size=256 \
            --lr=0.001 \
            --epochs=30 \
            --scheduler=cosine
        ;;
    2)
        echo "📋 方案2: 头部微调（只训练FC层）"
        echo "学习率: 0.001"
        echo "Batch Size: 256"
        echo "Epochs: 30"
        echo "冻结: 卷积层"
        echo ""
        python finetune-gpu.py \
            --model-path=../../../model/mnist_cnn3.pt \
            --batch-size=256 \
            --lr=0.001 \
            --epochs=30 \
            --freeze-conv \
            --scheduler=cosine
        ;;
    3)
        echo "📋 方案3: 深度微调（更激进）"
        echo "学习率: 0.002"
        echo "Batch Size: 512"
        echo "Epochs: 50"
        echo "Scheduler: Cosine"
        echo ""
        python finetune-gpu.py \
            --model-path=../../../model/mnist_cnn3.pt \
            --batch-size=512 \
            --lr=0.002 \
            --epochs=50 \
            --scheduler=cosine \
            --min-lr=0.00005
        ;;
    4)
        echo "📋 方案4: 保守微调（小心翼翼）"
        echo "学习率: 0.0005"
        echo "Batch Size: 256"
        echo "Epochs: 30"
        echo "Dropout: 0.2"
        echo ""
        python finetune-gpu.py \
            --model-path=../../../model/mnist_cnn3.pt \
            --batch-size=256 \
            --lr=0.0005 \
            --epochs=30 \
            --scheduler=cosine \
            --min-lr=0.00001 \
            --dropout=0.2
        ;;
    test)
        echo "📋 测试模型加载"
        python test-model.py --model-path=../../../model/mnist_cnn3.pt
        ;;
    *)
        echo "❌ 未知的方案: $PLAN"
        echo ""
        echo "可用方案:"
        echo "  1 - 轻度微调（推荐）"
        echo "  2 - 头部微调（只训练FC层）"
        echo "  3 - 深度微调（更激进）"
        echo "  4 - 保守微调（小心翼翼）"
        echo "  test - 测试模型加载"
        echo ""
        echo "使用方法: bash run_finetune.sh [方案号]"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "✅ 训练完成！"
echo "======================================"
echo ""
echo "💡 查看结果:"
echo "  tensorboard --logdir=logs_finetune"
echo ""
echo "📁 模型保存位置:"
echo "  ../data/mnt/mnist_cnn_finetuned.pt"
echo ""
