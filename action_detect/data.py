import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# 加载出来一个数据集
class PoseDataSet(Dataset):
    # 包含文件的根目录以及是训练集还是测试集
    def __init__(self, root, is_train = True):
        super(PoseDataSet, self).__init__()
        # 初始化一个数据集数据结构
        self.dataset = []
        # 根据参数选择子目录
        sub_dir = "train" if is_train else "test"
        # 在两个目录中分别读取数据
        for tag in os.listdir(f"{root}/{sub_dir}"):
            file_dir = f"{root}/{sub_dir}/{tag}"
            for img_file in os.listdir(file_dir):
                img_path = f"{file_dir}/{img_file}"
                if tag == 'HeadDown':
                    self.dataset.append((img_path, 0))
                else:
                    self.dataset.append((img_path, 1))
                # 输出图片数据,图片存储为一个链表的形式，可以直接调用
                # print(self.dataset)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)  # 以灰度图形式读数据
        img = img / 255.0  # 把数据转成 [0, 1] 之间的数据

        # 变为tensor类型存储起来，并且第一个位置是通道数
        transf = transforms.ToTensor()
        img = transf(img)  # tensor数据格式是torch(C,H,W)
        tag_one_hot = np.zeros(2)
        tag_one_hot[int(data[1])] = 1
        # 返回原始的图像数据以及标签
        return np.float32(img), np.float32(tag_one_hot)

if __name__ == '__main__':
    dataset = PoseDataSet('D:/OneDrive/桌面/Openpose/data')
    # gittem函数之后将图片改为128*128的函数
    print(dataset[0][0].shape)