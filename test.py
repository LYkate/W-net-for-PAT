import pandas
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import glob
import numpy as np
import cv2
from PATmodel import *

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = PATModel(in_channels=1, out_channels=1)
    # 将网络拷贝到deivce中
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)  # 包装为并行风格模型
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有信号数据路径
    tests_path = glob.glob(r"test/*.xlsx")
    # 遍历所有信号数据
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取信号数据
        input_ex=pd.read_excel(test_path, header=None)
        input_signal = torch.tensor(input_ex.values)
        #input_signal = torch.unsqueeze(input_signal, 0)
        # 转为batch为1，通道为1，大小为64*640的数组
        input_tensor = torch.reshape(input_signal, (1, 1, 64, 640))
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        input_tensor = input_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(input_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 保存图片
        cv2.imwrite(save_res_path, pred*255)
