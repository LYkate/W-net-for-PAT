import numpy as np
import pandas
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
from matplotlib import pyplot as plt
import torch.nn.functional as F
import cv2

class MyDataset(Dataset): #继承了Dataset类
    def __init__(self,input_root,lable_root):
        #分别读取输入/标签的路径信息
        self.input_root=input_root
        self.input_files=os.listdir(input_root)

        self.lable_root = lable_root
        self.lable_files = os.listdir(lable_root)

        self.lable_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize([64, 64]),
            torchvision.transforms.ToTensor()
        ])
    def __len__(self):
        #获取数据集大小
        return len(self.input_files)
    def __getitem__(self, index):
       input_signal_path=os.path.join(self.input_root,self.input_files[index])
       input_ex=pd.read_excel(input_signal_path, header=None)

       lable_img_path = os.path.join(self.lable_root, self.lable_files[index])
       lable_img = Image.open(lable_img_path)

       input_signal=torch.tensor(input_ex.values)
       input_signal=torch.unsqueeze(input_signal, 0)
       lable_img=self.lable_transforms(lable_img)

       return (input_signal,lable_img)
