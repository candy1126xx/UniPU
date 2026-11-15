"""
模型 v2：6层MLP（任务书要求，与PUe原论文一致）。

架构：784-256-256-128-128-64-64-1（6隐藏层+输出层）
"""
import torch
import torch.nn as nn


class MLP6_v2(nn.Module):
    """6层MLP（任务书v1.1配置，PUe原论文backbone）。"""
    def __init__(self, input_dim=784, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        return self.fc_out(x)


class MLP6_Small(nn.Module):
    """小容量6层MLP（任务书v3.9配置，~60K参数，欠拟合区间）。
    架构：784-64-64-32-32-16-1
    """
    def __init__(self, input_dim=784, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return self.fc_out(x)


class LinearModel(nn.Module):
    """线性模型（~0.8K参数，极端欠拟合）。
    架构：784-1
    """
    def __init__(self, input_dim=784, output_dim=1):
        super().__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.fc_out(x)


class MLP2_Tiny(nn.Module):
    """2层小MLP（~12.6K参数，极端欠拟合）。
    架构：784-16-1
    """
    def __init__(self, input_dim=784, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc_out = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc_out(x)


class MLP2(nn.Module):
    """2层MLP（E0实验标准配置，~6K参数 for Adult 93维）。
    架构：input-64-1
    """
    def __init__(self, input_dim=784, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc_out = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc_out(x)


class MLP2_Generic(nn.Module):
    """通用2层MLP，支持自定义隐藏层维度。
    架构：input-hidden_dim-1
    """
    def __init__(self, input_dim=784, output_dim=1, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc_out(x)


class MLP6_MultiClass(nn.Module):
    """6层MLP多类输出版本（FedPU用，10类输出）。"""
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        return self.fc_out(x)


def make_model_v2(model_type='mlp6', input_dim=784, output_dim=1, device='cuda'):
    if model_type == 'linear':
        return LinearModel(input_dim=input_dim, output_dim=output_dim).to(device)
    if model_type == 'mlp2_tiny':
        return MLP2_Tiny(input_dim=input_dim, output_dim=output_dim).to(device)
    if model_type == 'mlp2':
        return MLP2(input_dim=input_dim, output_dim=output_dim).to(device)
    # 通用2层MLP：mlp2_h{dim}，如 mlp2_h16, mlp2_h32, mlp2_h64, mlp2_h128, mlp2_h256
    if model_type.startswith('mlp2_h'):
        hidden_dim = int(model_type.split('_h')[1])
        return MLP2_Generic(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)
    if model_type == 'mlp6_small':
        return MLP6_Small(input_dim=input_dim, output_dim=output_dim).to(device)
    if model_type == 'mlp6_multiclass':
        return MLP6_MultiClass(input_dim=input_dim, num_classes=output_dim).to(device)
    if model_type == 'cnn13':
        return CNN13(output_dim=output_dim).to(device)
    if model_type == 'cnn13_multiclass':
        return CNN13_MultiClass(num_classes=output_dim).to(device)
    if model_type == 'small_cnn':
        return SmallCNN(output_dim=output_dim, input_dim=input_dim).to(device)
    if model_type.startswith('resnet1d'):
        return ResNet1D(input_dim=input_dim, output_dim=output_dim).to(device)
    return MLP6_v2(input_dim=input_dim, output_dim=output_dim).to(device)


class SmallCNN(nn.Module):
    """3层小CNN（~102K参数，无BN）。

    同时支持三种输入：
      input_dim=784  → 1×28×28（MNIST）
      input_dim=3072 → 3×32×32（CIFAR-10）
    架构：3 Conv(ReLU+MaxPool) + GAP + 2FC
    去掉BN：避免联邦聚合后running stats不一致导致的训练恶化
    """
    def __init__(self, output_dim=1, input_dim=3072):
        super().__init__()
        if input_dim == 784:
            self.in_ch, self.in_size = 1, 28
        elif input_dim == 3072:
            self.in_ch, self.in_size = 3, 32
        else:
            self.in_ch = 1
            self.in_size = int(round((input_dim / (16 * 16)) ** 0.5)) \
                if input_dim >= 256 else 28
        self.conv1 = nn.Conv2d(self.in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(x.size(0), self.in_ch, self.in_size, self.in_size)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc_out(x)


class CNN13(nn.Module):
    """13层CNN（任务书v1.1配置，PUe原论文backbone for CIFAR-10）。

    4个卷积块（每块2个卷积层+BN+ReLU+MaxPool）+ AdaptiveAvgPool(1x1) + 2个FC层
    = 8卷积 + 2FC = 10权重层 + BN层
    """
    def __init__(self, output_dim=1):
        super().__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Conv Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Conv Block 3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Conv Block 4
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        # FC layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(x.size(0), 3, 32, 32)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)


class CNN13_MultiClass(nn.Module):
    """13层CNN多类输出版本（FedPU用，10类输出）。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(x.size(0), 3, 32, 32)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)


# ============================================================
# 1D 残差 CNN（用于一维序列输入）
# ============================================================
class _BasicBlock1d(nn.Module):
    """1D 残差块。用 GroupNorm 而非 BN：联邦聚合下 BN 的 running stats
    会随各客户端数据漂移，聚合后反而伤性能（与 SmallCNN 去掉 BN 同理）。"""

    def __init__(self, in_ch, out_ch, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch))

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """1D ResNet（输入 (N, n_leads, length) 的一维序列）。

    输入 (N, input_dim) 或 (N, n_leads, length)，input_dim = n_leads*length。
    结构：stem(k=7,s=2) + MaxPool + 4 stage × 2 block + GAP + FC（约 1M 参数）。
    """

    def __init__(self, input_dim=12000, output_dim=1, n_leads=12,
                 width=(32, 64, 128, 256), blocks=(2, 2, 2, 2)):
        super().__init__()
        assert input_dim % n_leads == 0, (input_dim, n_leads)
        self.n_leads = n_leads
        self.length = input_dim // n_leads
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, width[0], 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, width[0]), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        stages, in_ch = [], width[0]
        for s, (w, nb) in enumerate(zip(width, blocks)):
            for b in range(nb):
                stages.append(_BasicBlock1d(
                    in_ch, w, 2 if (b == 0 and s > 0) else 1))
                in_ch = w
        self.stages = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(width[-1], 128)
        self.fc_out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(x.size(0), self.n_leads, self.length)
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc_out(x)
