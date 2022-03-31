'''
-*- encoding: UTF-8 -*-
Description      :init save and load model
Time             :2021/04/26 11:11:26
Author           :chentao
Version          :1.0
'''


import torch
import time
import os
from torch import nn
from torchvision import models


class Model:
    def __init__(self, name):
        self.name = name
        print("use model named:{}".format(name))
        # 加载模型，使用预训练模型
        self.backbone = models.__dict__[name](pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 2)
        )
        # self.backbone.fc=nn.Linear(2048,2)

    def load(self, path):
        self.backbone.load_state_dict(torch.load(path))

    def save(self, path):
        print("model are saved in:{}".format(path))
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(self.backbone.state_dict(), time.strftime(
            path + self.name + '-' + '%Y-%m-%d.pth'))
