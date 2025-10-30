"""
Fashion MNIST 增量训练脚本 - 基于预训练模型
从保存的checkpoint继续训练，支持调整学习率和优化策略
"""

from __future__ import print_function

import argparse
import os
from datetime import datetime
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))


# 卷积神经网络模型（与原始相同）
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


def train(args, model, device, train_loader, test_loader, optimizer, epoch, writer, early_stopping):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), niter)

        if args.val_interval > 0 and batch_idx > 0 and batch_idx % args.val_interval == 0:
            print(f'\n=== 中间验证 (Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}) ===')
            val_acc, val_loss = test(args, model, device, test_loader, writer, epoch, batch_idx)

            if early_stopping.check_and_save(val_loss, val_acc, model):
                print(f'💾 保存新的最佳模型 (验证损失: {val_loss:.4f}, 验证精度: {val_acc:.4f})')

            model.train()

    avg_train_loss = train_loss / num_batches
    return avg_train_loss


def test(args, model, device, test_loader, writer, epoch, batch_idx=None):
    """在测试集上评估模型"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = test_loss / len(test_loader.dataset)
    accuracy = float(correct) / len(test_loader.dataset)
    print('\nValidation Loss: {:.4f}, accuracy={:.4f}\n'.format(val_loss, accuracy))

    if batch_idx is not None:
        step = epoch * 10000 + batch_idx
        writer.add_scalar('accuracy_intra_epoch', accuracy, step)
        writer.add_scalar('val_loss_intra_epoch', val_loss, step)
    else:
        writer.add_scalar('accuracy', accuracy, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

    return accuracy, val_loss


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, verbose=False, delta=0.0001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_val_loss = None
        self.best_val_acc = None
        self.best_model_state = None
        self.early_stop = False
        self.val_count = 0

    def check_and_save(self, val_loss, val_acc, model):
        self.val_count += 1
        saved = False

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            saved = True
            if self.verbose:
                print(f'✅ 初始化最佳模型 (损失: {val_loss:.4f}, 精度: {val_acc:.4f})')
        elif val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            saved = True
            if self.verbose:
                print(f'✅ 验证损失降低到 {val_loss:.4f} (精度: {val_acc:.4f})')
        else:
            self.counter += 1
            if self.verbose:
                print(f'⚠️  连续 {self.counter}/{self.patience} 次验证无改进 (最佳损失: {self.best_val_loss:.4f})')

        return saved

    def __call__(self, val_loss, val_acc, model):
        self.check_and_save(val_loss, val_acc, model)

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f'🛑 早停触发！连续 {self.counter} 次验证无改进')

        return self.early_stop


def create_scheduler(optimizer, args):
    """创建学习率调度器"""
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    elif args.scheduler == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'linear':
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.95)
    else:
        return None

    return scheduler


def main():
    # ========== 1. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='PyTorch MNIST Finetune Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate for finetune (default: 0.001) - 比初始学习率小得多')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cosine', metavar='S',
                        choices=['none', 'step', 'exponential', 'cosine', 'linear', 'plateau'],
                        help='learning rate scheduler (default: cosine)')
    parser.add_argument('--scheduler-step', type=int, default=10, metavar='N',
                        help='step size for StepLR scheduler (default: 10)')
    parser.add_argument('--scheduler-gamma', type=float, default=0.5, metavar='G',
                        help='gamma for StepLR/ExponentialLR scheduler (default: 0.5)')
    parser.add_argument('--min-lr', type=float, default=0.00001, metavar='LR',
                        help='minimum learning rate (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables Mac GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging')
    parser.add_argument('--val-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before validation (0 = only at end of epoch)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='../data', help='dataset directory')
    parser.add_argument('--save-model-dir', type=str, default='../data/mnt', help='save directory')
    parser.add_argument('--model-path', type=str, required=True, metavar='PATH',
                        help='path to pretrained model (required)')
    parser.add_argument('--dir', default='logs_finetune', metavar='L',
                        help='directory where summary logs are stored')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='D',
                        help='dropout rate (default: 0.3)')
    parser.add_argument('--freeze-conv', action='store_true', default=False,
                        help='freeze convolutional layers, only finetune FC layers')
    parser.add_argument('--freeze-bn', action='store_true', default=False,
                        help='freeze batch norm layers')

    args = parser.parse_args()

    # ========== 2. 检查GPU设备 ==========
    use_cuda = False
    use_mps = False

    if not args.no_cuda:
        if torch.cuda.is_available():
            use_cuda = True
            print('Using CUDA')
        elif torch.backends.mps.is_available() and not args.no_mps:
            use_mps = True
            print('Using Mac GPU (MPS)')
        else:
            print('GPU not available, using CPU')
    else:
        print('GPU disabled, using CPU')

    # ========== 3. 初始化TensorBoard ==========
    writer = SummaryWriter(args.dir)

    # ========== 4. 设置随机种子 ==========
    torch.manual_seed(args.seed)

    # ========== 5. 设置计算设备 ==========
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ========== 6. 数据加载器配置 ==========
    if use_cuda:
        kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True}
    elif use_mps:
        kwargs = {'num_workers': 6, 'pin_memory': False, 'persistent_workers': True}
    else:
        kwargs = {'num_workers': 2}

    # 训练集数据加载器（轻度数据增强，避免过度增强）
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.dataset, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(28, padding=2),  # 轻度裁剪
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),  # 轻度旋转
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # 测试集数据加载器
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.dataset, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # ========== 7. 创建模型 ==========
    model = Net(dropout_rate=args.dropout).to(device)

    # ========== 8. 加载预训练模型 ==========
    if not os.path.exists(args.model_path):
        print(f'❌ 错误: 模型文件不存在: {args.model_path}')
        return

    print(f'📂 加载预训练模型: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print('✅ 模型加载成功')

    # ========== 9. 冻结部分层（可选） ==========
    if args.freeze_conv:
        print('🔒 冻结卷积层，只微调FC层')
        for name, param in model.named_parameters():
            if 'conv' in name or 'bn' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    if args.freeze_bn:
        print('🔒 冻结BatchNorm层')
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()  # 设置为eval模式，不更新统计信息

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'📊 总参数: {total_params:,}, 可训练参数: {trainable_params:,}')

    # ========== 10. 初始化优化器 ==========
    # 注意：使用较小的学习率进行微调
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-2
    )

    # ========== 11. 初始化学习率调度器 ==========
    scheduler = create_scheduler(optimizer, args)
    if scheduler is not None:
        print(f'使用 {args.scheduler} 调度器')
    else:
        print('不使用学习率调度器，固定学习率')

    # ========== 12. 初始化早停机制 ==========
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.0001)

    # ========== 13. 训练循环 ==========
    print("========== 开始增量训练 ==========")
    print(f"预训练模型: {args.model_path}")
    print(f"学习率: {args.lr} (微调用 - 较小)")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"开始时间: {datetime.now().strftime('%y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, test_loader, optimizer, epoch, writer, early_stopping)

        print(f'\n=== Epoch {epoch} 结束验证 ===')
        val_acc, val_loss = test(args, model, device, test_loader, writer, epoch)

        writer.add_scalar('train_loss_avg', train_loss, epoch)

        current_lrs = [group['lr'] for group in optimizer.param_groups]
        if len(current_lrs) == 1:
            print('Epoch {} learning rate: {:.6f}'.format(epoch, current_lrs[0]))
            writer.add_scalar('learning_rate', current_lrs[0], epoch)
        else:
            lr_values = ', '.join(['{:.6f}'.format(lr) for lr in current_lrs])
            print('Epoch {} learning rates: {}'.format(epoch, lr_values))
            for index, lr in enumerate(current_lrs):
                writer.add_scalar('learning_rate/group_{}'.format(index), lr, epoch)

        # 更新学习率
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 检查早停条件
        if early_stopping(val_loss, val_acc, model):
            print(f'\n{"="*40}')
            print(f'🛑 早停触发！在epoch {epoch}停止训练')
            print(f'最佳验证损失: {early_stopping.best_val_loss:.4f}')
            print(f'最佳验证精度: {early_stopping.best_val_acc:.4f}')
            print(f'{"="*40}\n')
            break

    print("========== 训练完成 ==========")
    print(f"结束时间: {datetime.now().strftime('%y-%m-%d %H:%M:%S')}")

    # ========== 14. 保存模型 ==========
    if args.save_model:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        if early_stopping.best_model_state is not None:
            print(f'\n💾 保存最佳微调模型')
            print(f'   验证损失: {early_stopping.best_val_loss:.4f}')
            print(f'   验证精度: {early_stopping.best_val_acc:.4f}')
            torch.save(early_stopping.best_model_state,
                      os.path.join(args.save_model_dir, "mnist_cnn_finetuned.pt"))
        else:
            print(f'\n💾 保存当前模型')
            torch.save(model.state_dict(),
                      os.path.join(args.save_model_dir, "mnist_cnn_finetuned.pt"))

    writer.close()


if __name__ == '__main__':
    main()
