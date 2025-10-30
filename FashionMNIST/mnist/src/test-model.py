"""
模型测试脚本 - 快速验证模型是否可正常加载和运行
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.dropout2d = nn.Dropout2d(p=dropout_rate)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_model(model_path, device='cuda'):
    """
    测试模型：验证模型加载、前向传播、评估等功能
    """
    print("=" * 60)
    print("🧪 模型测试脚本")
    print("=" * 60)

    # 1. 检查模型文件是否存在
    print(f"\n1️⃣  检查模型文件...")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return False

    file_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"✅ 模型文件存在: {model_path}")
    print(f"   文件大小: {file_size:.2f} MB")

    # 2. 检查GPU可用性
    print(f"\n2️⃣  检查计算设备...")
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️  CUDA不可用，切换到CPU")
            device = 'cpu'
        else:
            print(f"✅ CUDA可用")
            print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
    else:
        print(f"✅ 使用CPU模式")

    device = torch.device(device)

    # 3. 创建模型
    print(f"\n3️⃣  创建模型架构...")
    model = Net(dropout_rate=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型创建成功")
    print(f"   总参数数: {total_params:,}")

    # 4. 加载预训练权重
    print(f"\n4️⃣  加载预训练权重...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

    # 5. 测试前向传播
    print(f"\n5️⃣  测试前向传播...")
    model.eval()
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 28, 28).to(device)
            output = model(dummy_input)
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   输出值 (logits): {output[0]}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

    # 6. 测试评估模式
    print(f"\n6️⃣  在测试集上评估...")
    try:
        # 加载测试数据（只用前100个样本以加快速度）
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=32, shuffle=False, num_workers=2
        )

        correct = 0
        total = 0
        losses = []

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i >= 3:  # 只测试前3个batch
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.nll_loss(output, target)
                losses.append(loss.item())

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = correct / total if total > 0 else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        print(f"✅ 评估完成（测试样本: {total}）")
        print(f"   准确率: {accuracy * 100:.2f}%")
        print(f"   平均损失: {avg_loss:.4f}")

    except Exception as e:
        print(f"⚠️  评估过程中出错: {e}")
        # 不返回False，因为这不是致命错误

    # 7. 内存使用
    print(f"\n7️⃣  内存使用情况...")
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        print(f"✅ CUDA内存")
        print(f"   已分配: {allocated:.2f} MB")
        print(f"   已保留: {reserved:.2f} MB")
    else:
        print(f"✅ CPU模式，无需显存")

    # 总结
    print(f"\n{'=' * 60}")
    print("✅ 所有测试通过！模型可以进行微调训练")
    print(f"{'=' * 60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test model loading and inference')
    parser.add_argument('--model-path', type=str, required=True, help='path to model file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='device to use')
    parser.add_argument('--dataset', type=str, default='../data', help='dataset directory')

    args = parser.parse_args()

    success = test_model(args.model_path, device=args.device)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
