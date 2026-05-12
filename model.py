'''ResNet在PyTorch中的实现
对于预激活ResNet，请参考'preact_resnet.py'
参考文献:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    基础残差块，用于ResNet18和ResNet34
    包含两个3x3卷积层，适用于较浅的网络
    """
    expansion = 1  # 通道扩展倍数，基础块为1

    def __init__(self, in_planes, planes, stride=1):
        """
        初始化基础残差块
        Args:
            in_planes: 输入通道数
            planes: 输出通道数
            stride: 步长，用于下采样
        """
        super(BasicBlock, self).__init__()
        
        # 第一个3x3卷积层
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化
        
        # 第二个3x3卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 批归一化

        # 跳跃连接（shortcut connection）
        self.shortcut = nn.Sequential()
        # 当步长不为1或输入输出通道数不匹配时，需要调整跳跃连接
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积调整维度
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            输出张量
        """
        # 第一个卷积 -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二个卷积 -> BN
        out = self.bn2(self.conv2(out))
        # 残差连接：将输入通过跳跃连接加到输出上
        out += self.shortcut(x)
        # 最终激活
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    瓶颈残差块，用于ResNet50、ResNet101和ResNet152
    包含三个卷积层：1x1 -> 3x3 -> 1x1，适用于更深的网络
    """
    expansion = 4  # 通道扩展倍数，瓶颈块为4

    def __init__(self, in_planes, planes, stride=1):
        """
        初始化瓶颈残差块
        Args:
            in_planes: 输入通道数
            planes: 中间层通道数
            stride: 步长，用于下采样
        """
        super(Bottleneck, self).__init__()
        
        # 第一个1x1卷积，降维
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二个3x3卷积，主要计算
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 第三个1x1卷积，升维
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        # 当步长不为1或输入输出通道数不匹配时，需要调整跳跃连接
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积调整维度
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            输出张量
        """
        # 第一个卷积 -> BN -> ReLU（降维）
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二个卷积 -> BN -> ReLU（主要计算）
        out = F.relu(self.bn2(self.conv2(out)))
        # 第三个卷积 -> BN（升维）
        out = self.bn3(self.conv3(out))
        # 残差连接
        out += self.shortcut(x)
        # 最终激活
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet主网络结构
    包含四个残差层组，每组包含多个残差块
    """
    def __init__(self, block, num_blocks, num_classes=10, return_features=False):
        """
        初始化ResNet
        Args:
            block: 残差块类型（BasicBlock或Bottleneck）
            num_blocks: 每层的残差块数量列表 [layer1, layer2, layer3, layer4]
            num_classes: 分类类别数，默认为10（如CIFAR-10）
            return_features: 是否返回特征
        """
        super(ResNet, self).__init__()
        self.in_planes = 64  # 当前输入通道数
        self.return_features = return_features

        # 初始卷积层：将3通道RGB图像转为64通道特征图
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 四个残差层组
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # 64通道
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 128通道，下采样
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 256通道，下采样
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 512通道，下采样
        
        # 自适应平均池化，将特征图池化为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 最终分类层
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        构建残差层
        Args:
            block: 残差块类型
            planes: 输出通道数
            num_blocks: 该层中残差块的数量
            stride: 第一个残差块的步长
        Returns:
            包含多个残差块的Sequential模块
        """
        # 步长列表：第一个块使用指定步长，其余块步长为1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            # 添加残差块
            layers.append(block(self.in_planes, planes, stride))
            # 更新输入通道数（下一个块的输入通道数）
            self.in_planes = planes * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播 - 完整分类
        Args:
            x: 输入图像张量 [batch_size, 3, height, width]
        Returns:
            分类logits [batch_size, num_classes]
        """
        # 初始卷积 -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 通过四个残差层组
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 全局平均池化
        out = self.avgpool(out)
        
        # 展平为一维向量
        features = out.view(out.size(0), -1)
        
        # 分类层
        out = self.linear(features)
        
        if self.return_features:
            return out, features
        else:
            return out

    def create_emb(self, x):
        """
        创建特征嵌入（不经过最终分类层）
        用于特征提取或迁移学习
        Args:
            x: 输入图像张量
        Returns:
            特征嵌入向量 [batch_size, 512*expansion]
        """
        # 初始卷积 -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 通过四个残差层组
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 全局平均池化
        out = self.avgpool(out)
        
        # 展平为一维向量（特征嵌入）
        out = out.view(out.size(0), -1)
        return out


# 预定义的ResNet模型构造函数

def ResNet18(num_classes=10, return_features=False):
    """
    构建ResNet-18模型
    结构：[2, 2, 2, 2] - 每层包含2个基础残差块
    总共18层（2*2*4 + 2 = 18层）
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, return_features=return_features)


def ResNet34(num_classes=10, return_features=False):
    """
    构建ResNet-34模型
    结构：[3, 4, 6, 3] - 每层包含不同数量的基础残差块
    总共34层（(3+4+6+3)*2 + 2 = 34层）
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, return_features=return_features)


def ResNet50(num_classes=10, return_features=False):
    """
    构建ResNet-50模型
    结构：[3, 4, 6, 3] - 每层包含不同数量的瓶颈残差块
    总共50层（(3+4+6+3)*3 + 2 = 50层）
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, return_features=return_features)


def ResNet101(num_classes=10, return_features=False):
    """
    构建ResNet-101模型
    结构：[3, 4, 23, 3] - 第三层包含23个瓶颈残差块
    总共101层（(3+4+23+3)*3 + 2 = 101层）
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, return_features=return_features)


def ResNet152(num_classes=10, return_features=False):
    """
    构建ResNet-152模型
    结构：[3, 8, 36, 3] - 最深的ResNet变体
    总共152层（(3+8+36+3)*3 + 2 = 152层）
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, return_features=return_features)


def test():
    """
    测试函数：创建ResNet18并测试前向传播
    """
    net = ResNet18()
    # 创建随机输入：批量大小1，3通道，32x32像素（CIFAR-10尺寸）
    y = net(torch.randn(1, 3, 32, 32))
    print(f"输出张量形状: {y.size()}")  # 应该是 [1, 10] 对于10分类

# 取消注释以运行测试
# test()