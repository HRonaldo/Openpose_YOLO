import torch
from torch import nn, randn, exp, sum
from torchvision.models import resnet18


# 存储神经网络

# 最简单的神经网络，从16384的数据转换为两个集合，属于二分类问题
class NetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(randn(16384, 2))
    def forward(self, x):
        h = x @ self.W
        # soft max
        h = exp(h)
        z = sum(h, dim=1, keepdim=True)  # 保持梯度
        return h / z



# 升级之后的网络，总的来说使用了两个线性层以及一个激活函数，是一个隐藏层的数据结构
class NetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)


# 最终优化的模型
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # 2是二分类任务的输出维度


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.resnet = resnet18(pretrained=True)
        # 修改输入通道数为1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 2)  # 2是二分类任务的输出维度

    def forward(self, x):
        x = self.resnet(x)
        return x

if __name__ == '__main__':
    x = torch.ones((100,1,128,128))
    Geek_Huang = BinaryClassifier()
    print(Geek_Huang.forward(x))
