import numpy as np
import torch.cuda
from torch import from_numpy, argmax
from torchvision.transforms import transforms

# 用于在runOpenpose中运行这段代码，是模型对外的接口

DEVICE = "gpu"
if torch.cuda.is_available():
    DEVICE = "gpu"
else:
    DEVICE = "cpu"

def action_detect(net,pose,crown_proportion):
    maxHeight = pose.keypoints.max()
    minHeight = pose.keypoints.min()
    img = pose.img_pose
    img = img / 255  # 把数据转成[0,1]之间的数据
    transf = transforms.ToTensor()
    img = transf(img)  # tensor数据格式是torch(C,H,W)
    img = np.float32(img)
    img = from_numpy(img[None,:]).cuda()
    # 神经网络计算的结结果值
    predect = net(img)
    action_id = int(argmax(predect,dim=1).cpu().detach().item())

    # crown非常关键
    possible_rate = predect[:,action_id]
    possible_rate = possible_rate.cpu().detach().numpy()[0]
    possible_rate = possible_rate.item()
    if possible_rate > 0.75:
        pose.pose_action = 'HeadDown'
        if possible_rate > 1:
            possible_rate = 1
        pose.action_fall = possible_rate
        pose.action_normal = 1-possible_rate
    else:
        pose.pose_action = 'normal'
        if possible_rate >= 0.5:
            pose.action_fall = 1-possible_rate
            pose.action_normal = possible_rate
        else:
            pose.action_fall = possible_rate
            pose.action_normal = 1 - possible_rate
    return pose