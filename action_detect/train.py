import torch
from action_detect.data import *
from action_detect.net import *
from torch.utils.data import DataLoader
from torch import optim
import time

# 训练自己的模型

if torch.cuda.is_available():
    print(1)
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class Train:
    def __init__(self, root):
        # 加载训练数据
        self.train_dataset = PoseDataSet(root, True)
        self.train_dataLoader = DataLoader(self.train_dataset, batch_size=10, shuffle=True)
        # 加载测试数据
        self.test_dataset = PoseDataSet(root, False)
        self.test_dataLoader = DataLoader(self.test_dataset, batch_size=10, shuffle=True)

        # 创建模型
        self.net = CNN()
        # 加载已训练的数据
        # self.net.load_state_dict(torch.load("D:/OneDrive/桌面/Openpose/action_detect/checkPoint/action.pth"))

        self.net.to(DEVICE)  # 使用GPU进行训练


        # 定义优化器
        self.opt = optim.Adam(self.net.parameters(), lr=0.001)  # 加强版梯度下降法,SGD 普通梯度下降法

    # 启动训练
    def __call__(self):
        for epoch in range(20):
            train_sum_loss = 0
            test_sum_loss = 0

            # 训练集进行训练
            self.net.train()  # 表明在训练环境下进行
            for i, (imgs, tags) in enumerate(self.train_dataLoader):
                # 训练集添加到GPU
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)

                train_y = self.net(imgs)
                loss = torch.mean((tags - train_y) ** 2)

                self.opt.zero_grad()  # 清梯度
                loss.backward()  # 反向传播
                self.opt.step()  # 记录步数

                train_sum_loss += loss.item()
            train_avg_loss = train_sum_loss / len(self.train_dataLoader)

            # 测试集用于测试
            self.net.eval()  # 标明在测试环境下
            for i, (imgs, tags) in enumerate(self.test_dataLoader):
                # 测试集添加到GPU
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)
                test_y = self.net(imgs)
                loss = torch.mean((tags - test_y) ** 2)
                test_sum_loss += loss.item()
            test_avg_loss = test_sum_loss / len(self.test_dataLoader)
            print("第", epoch, "轮", "训练集平均损失值：", train_avg_loss, "测试集平均损失值：", test_avg_loss)

            # 添加时间戳
            now_time = int(time.time())
            timeArray = time.localtime(now_time)
            str_time = time.strftime("%Y-%m-%d-%H:%M:%S", timeArray)

            # 保存训练好的参数
            torch.save(self.net, f"./checkPoint/action.pth")


if __name__ == '__main__':
    train = Train('D:/OneDrive/桌面/Openpose/data')
    train()
