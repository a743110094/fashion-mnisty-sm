"""
Fashion MNIST 训练脚本 - 支持多种GPU后端
支持设备：CUDA (NVIDIA GPU) > MPS (Mac GPU) > CPU
作者备注：Mac GPU(MPS)需要PyTorch>=1.12.0，且M1/M2/M3等Apple Silicon芯片
"""

from __future__ import print_function

import argparse  # 命令行参数解析
import os
from datetime import datetime  # 时间戳记录
from tensorboardX import SummaryWriter  # TensorBoard 可视化日志
from torchvision import datasets, transforms  # 图像数据集和变换
import torch
import torch.distributed as dist  # 分布式训练支持
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数（激活函数、损失函数等）
import torch.optim as optim  # 优化器
import torch.optim.lr_scheduler as lr_scheduler  # 学习率调度器

# 获取分布式训练的全局进程数，默认为 1（单机训练）
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))


# 卷积神经网络模型，用于图像分类
class Net(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Net, self).__init__()
        # 第一层卷积：输入通道=1(灰度图), 输出通道=32, 卷积核=5x5, 步长=1
        self.conv1 = nn.Conv2d(1, 64, 5, 1)
        # 第一层批量归一化（卷积层后）
        self.bn1 = nn.BatchNorm2d(64)

        # 第二层卷积：输入通道=64, 输出通道=128, 卷积核=3x3, 步长=1
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        # 第二层批量归一化（卷积层后）
        self.bn2 = nn.BatchNorm2d(128)

        # 第三层卷积：输入通道=128, 输出通道=256, 卷积核=3x3, 步长=1
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        # 第三层批量归一化（卷积层后）
        self.bn3 = nn.BatchNorm2d(256)


        # Dropout层：在卷积层后应用2D dropout
        self.dropout2d = nn.Dropout2d(p=dropout_rate)

        # 第一层全连接层：输入=256*1*1=256（经过多次池化后）
        # 计算过程：28x28 -> 24x24(conv1) -> 12x12(pool) -> 10x10(conv2) -> 5x5(pool)
        #        -> 3x3(conv3) -> 1x1(pool) -> fc1(256维)
        self.fc1 = nn.Linear(256*1*1, 256)
        # 全连接层批量归一化
        self.bn_fc = nn.BatchNorm1d(256)

        # Dropout层：在全连接层后应用dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # 输出层：输入=256, 输出=10（10个衣服分类）
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # 第一个卷积块：卷积 -> BatchNorm -> ReLU激活 -> 最大池化(2x2) -> Dropout
        x = self.conv1(x)  # 28x28 -> 24x24
        x = self.bn1(x)  # 批量归一化
        x = F.relu(x)  # ReLU激活
        x = F.max_pool2d(x, 2, 2)  # 24x24 -> 12x12
        x = self.dropout2d(x)  # 应用2D dropout（池化后）

        # 第二个卷积块：卷积 -> BatchNorm -> ReLU激活 -> 最大池化(2x2) -> Dropout
        x = self.conv2(x)  # 12x12 -> 10x10
        x = self.bn2(x)  # 批量归一化
        x = F.relu(x)  # ReLU激活
        x = F.max_pool2d(x, 2, 2)  # 10x10 -> 5x5
        x = self.dropout2d(x)  # 应用2D dropout（池化后）

        # 第三个卷积块：卷积 -> BatchNorm -> ReLU激活 -> 最大池化(2x2) -> Dropout
        x = self.conv3(x)  # 5x5 -> 3x3
        x = self.bn3(x)  # 批量归一化
        x = F.relu(x)  # ReLU激活
        x = F.max_pool2d(x, 2, 2)  # 3x3 -> 1x1
        x = self.dropout2d(x)  # 应用2D dropout（池化后）

        # 展平为一维向量：(batch_size, 256*1*1)
        x = x.view(-1, 256*1*1)

        # 全连接层 -> BatchNorm -> ReLU激活 -> Dropout
        x = self.fc1(x)
        x = self.bn_fc(x)  # 批量归一化
        x = F.relu(x)  # ReLU激活
        x = self.dropout(x)  # 应用dropout（隐藏层后）

        # 输出层（不带激活、不带dropout）
        x = self.fc2(x)

        # 返回log_softmax用于NLL损失函数
        return F.log_softmax(x, dim=1)

# 训练函数：对一个epoch的数据进行训练
def train(args, model, device, train_loader, optimizer, epoch, writer):
    # 设置模型为训练模式（启用dropout、batchnorm等）
    model.train()

    # 累计训练损失
    train_loss = 0
    num_batches = 0

    # 遍历训练集中的每一个batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移到指定设备（CPU或GPU）
        data, target = data.to(device), target.to(device)

        # 清空梯度（防止梯度累积）
        optimizer.zero_grad()

        # 前向传播：计算模型输出
        output = model(data)

        # 计算损失函数（Negative Log Likelihood Loss）
        loss = F.nll_loss(output, target)

        # 反向传播：计算梯度
        loss.backward()

        # 梯度裁剪：防止梯度爆炸（在Mac MPS上特别重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 累计损失
        train_loss += loss.item()
        num_batches += 1

        # 每隔log_interval个batch打印一次训练进度
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            # 记录损失到TensorBoard
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), niter)

    # 计算该epoch的平均训练损失
    avg_train_loss = train_loss / num_batches
    return avg_train_loss

# 测试函数：在测试集上评估模型性能
def test(args, model, device, test_loader, writer, epoch):
    # 设置模型为评估模式（禁用dropout、batchnorm等）
    model.eval()

    # 初始化测试损失和正确数
    test_loss = 0
    correct = 0

    # 在评估过程中禁用梯度计算（节省内存和计算量）
    with torch.no_grad():
        # 遍历测试集中的每一个batch
        for data, target in test_loader:
            # 将数据移到指定设备
            data, target = data.to(device), target.to(device)

            # 前向传播：计算模型输出
            output = model(data)

            # 累计测试损失（使用sum而不是mean，最后再除以总样本数）
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # 获取预测结果：取概率最大的类别索引
            pred = output.max(1, keepdim=True)[1]

            # 累计正确预测的样本数
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均测试损失（验证集损失）
    val_loss = test_loss / len(test_loader.dataset)

    # 计算并打印准确率
    accuracy = float(correct) / len(test_loader.dataset)
    print('\nValidation Loss: {:.4f}, accuracy={:.4f}\n'.format(val_loss, accuracy))

    # 记录准确率和验证损失到TensorBoard
    writer.add_scalar('accuracy', accuracy, epoch)
    writer.add_scalar('val_loss', val_loss, epoch)

    # 返回验证损失，供scheduler使用
    return val_loss


# 初始化学习率调度器
def create_scheduler(optimizer, args):
    """
    根据参数创建学习率调度器

    支持的调度器类型：
    - 'none': 不使用调度器，学习率保持不变
    - 'step': 每隔固定步数降低学习率 (StepLR)
    - 'exponential': 按指数衰减学习率 (ExponentialLR)
    - 'cosine': 使用余弦函数调度学习率 (CosineAnnealingLR)
    - 'linear': 线性衰减学习率 (LinearLR)
    - 'plateau': 当验证损失停止改进时降低学习率 (ReduceLROnPlateau) - 推荐！

    注意：ReduceLROnPlateau需要在每个epoch传入验证损失：scheduler.step(val_loss)
    其他调度器只需调用：scheduler.step()

    Args:
        optimizer: PyTorch 优化器实例
        args: 包含调度器参数的命令行参数对象

    Returns:
        scheduler 对象或 None
    """
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'step':
        # StepLR: 每 step_size 个 epoch 乘以 gamma
        # 例如：gamma=0.1, step_size=10 表示每10个epoch学习率乘以0.1
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    elif args.scheduler == 'exponential':
        # ExponentialLR: 每个 epoch 乘以 gamma
        # 例如：gamma=0.95 表示每个epoch学习率乘以0.95
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    elif args.scheduler == 'cosine':
        # CosineAnnealingLR: 使用余弦函数从初始LR衰减到最小LR
        # T_max 是周期（epoch数），after T_max 个 epoch 学习率将衰减到最小值
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'linear':
        # LinearLR: 线性衰减学习率
        # total_iters 是总迭代次数
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs)
    elif args.scheduler == 'plateau':
        # ReduceLROnPlateau: 当验证集的准确率没有提升时，将学习率衰减
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1)
    else:
        return None

    return scheduler


# 判断是否应该启用分布式训练
def should_distribute():
    # 检查分布式训练是否可用且进程数大于1
    return dist.is_available() and WORLD_SIZE > 1


# 判断分布式训练是否已初始化
def is_distributed():
    # 检查分布式训练是否可用且已初始化
    return dist.is_available() and dist.is_initialized()


# 主函数：训练流程的入口
def main():
    # ========== 1. 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=600, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9) - 注：使用Adam优化器时此参数不起作用')
    parser.add_argument('--scheduler', type=str, default='plateau', metavar='S',
                        choices=['none', 'step', 'exponential', 'cosine', 'linear', 'plateau'],
                        help='learning rate scheduler: none, step, exponential, cosine, linear, plateau (default: plateau)')
    parser.add_argument('--scheduler-step', type=int, default=10, metavar='N',
                        help='step size for StepLR scheduler (fault: 30)')
    parser.add_argument('--scheduler-gamma', type=float, default=0.5, metavar='G',
                        help='gamma for StepLR/ExponentialLR scheduler (default: 0.02)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPU training (CUDA and MPS)')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables Mac GPU (MPS) training, use CPU instead')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='../data', help='For dataset director')

    parser.add_argument('--save-model-dir', type=str, default='../data/mnt', help='For Saving directory')

    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='D',
                        help='dropout rate (default: 0.2)')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()

    # ========== 2. 检查并设置GPU设备 ==========
    # 优先级：CUDA > MPS (Mac) > CPU
    use_cuda = False
    use_mps = False

    if not args.no_cuda:
        if torch.cuda.is_available():
            use_cuda = True
            print('Using CUDA')
        elif torch.backends.mps.is_available() and not args.no_mps:
            # Mac GPU 支持 (Metal Performance Shaders)
            use_mps = True
            print('Using Mac GPU (MPS)')
        else:
            if torch.backends.mps.is_available() and args.no_mps:
                print('Mac GPU (MPS) is available but disabled by --no-mps flag')
            else:
                print('GPU not available, using CPU')
    else:
        print('GPU disabled by --no-cuda flag, using CPU')

    # ========== 3. 初始化TensorBoard日志写入器 ==========
    writer = SummaryWriter(args.dir)

    # ========== 4. 设置随机种子（保证可重复性） ==========
    torch.manual_seed(args.seed)

    # ========== 5. 设置计算设备 ==========
    # 根据可用的GPU类型选择设备
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ========== 6. 初始化分布式训练（如果需要） ==========
    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    # ========== 7. 数据加载器配置 ==========
    # 根据设备类型配置数据加载参数
    # CUDA: 启用多进程和内存锁定以加速数据传输
    # MPS: 不使用pin_memory（Mac GPU不支持），使用多进程加速
    # CPU: 基础多进程配置
    if use_cuda:
        kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    elif use_mps:
        # Mac GPU 不支持 pin_memory，但可以用多进程加速数据加载
        kwargs = {'num_workers': 6, 'pin_memory': False, 'persistent_workers': True}
    else:
        # CPU 模式
        kwargs = {'num_workers': 2}

    # 训练集数据加载器（含数据增强）
    # - FashionMNIST数据集自动下载
    # - 应用数据增强：随机旋转、偏移、亮度调整
    # - 转换为张量并进行标准化
    # - 打乱数据以增加泛化能力
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.dataset, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation(10),  # 随机旋转±10度
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移±10%
                        transforms.ColorJitter(brightness=0.2),  # 随机亮度调整±20%
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # 测试集数据加载器
    # - 不下载（如果已存在）
    # - 不打乱数据
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.dataset, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # ========== 8. 创建模型并移到设备 ==========
    model = Net(dropout_rate=args.dropout).to(device)

    # ========== 9. 启用分布式数据并行（如果需要） ==========
    # 注意：Mac MPS 目前不支持 DistributedDataParallel，仅支持单机训练
    if is_distributed():
        if use_mps:
            print('Warning: Mac GPU (MPS) does not support DistributedDataParallel yet.')
            print('Using single-machine training only.')
        else:
            Distributor = nn.parallel.DistributedDataParallel if use_cuda else nn.parallel.DistributedDataParallelCPU
            model = Distributor(model)

    # ========== 10. 初始化优化器 ==========
    # 使用Adam优化器：自适应学习率，特别适合深层网络
    # Adam = Adaptive Moment Estimation，结合了momentum和RMSprop的优点
    # weight_decay=1e-4 是L2正则化，防止权重过大（过拟合）
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ========== 10.5. 初始化学习率调度器 ==========
    # 根据参数初始化学习率调度器（可选）
    scheduler = create_scheduler(optimizer, args)
    if scheduler is not None:
        print('Using {} scheduler'.format(args.scheduler))
    else:
        print('No learning rate scheduler, using fixed learning rate')

    # ========== 11. 训练循环 ==========
    print("begin training: ", datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    for epoch in range(1, args.epochs + 1):
        # 在训练集上进行训练，获取平均训练损失
        train_loss = train(args, model, device, train_loader, optimizer, epoch, writer)

        # 在测试集上进行评估，获取验证损失
        val_loss = test(args, model, device, test_loader, writer, epoch)

        # 记录平均训练损失到TensorBoard
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

        # 在每个epoch后更新学习率（如果使用了调度器）
        if scheduler is not None:
            # ReduceLROnPlateau需要传入验证损失，其他调度器不需要
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

    print("end training: ", datetime.now().strftime('%y-%m-%d %H:%M:%S'))

    # ========== 12. 保存模型 ==========
    if (args.save_model):
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        torch.save(model.state_dict(), os.path.join(args.save_model_dir, "mnist_cnn.pt"))

if __name__ == '__main__':
    main()
