'''
-*- encoding: UTF-8 -*-
Description      :使用model去对训练集图片进行预测，用来同标注的结果对比
Time             :2021/06/23 12:04:01
Author           :chentao
Version          :1.0
'''

import torch
from pathlib import Path
from torch.utils import data
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
from xlwt.Workbook import Workbook
from model import Model
from torch.autograd import Variable
from torch import nn
from xlutils.copy import copy
import datetime
import os
from shutil import copy
import shutil

# 写入excel的数据   classes：图片名称，label标注标签
com_classes = []
com_label = []


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class preData(data.Dataset):
    def __init__(self, root):
        dir_path = Path(root)
        classes = []

        for category in dir_path.iterdir():
            if category.is_dir():
                classes.append(category.name)

        classes.sort()

        classes_list = []
        labels_list = []

        for index, name in enumerate(classes):
            class_path = dir_path / name
            if not class_path.is_dir():
                continue
            for img_path in class_path.glob('*.png'):
                classes_list.append(str(img_path))
                labels_list.append(index)

        classes_list, labels_list = zip(
            *sorted(zip(classes_list, labels_list)))

        global com_classes, com_label
        com_classes = classes_list
        com_label = labels_list

        self.data = classes_list
        self.label = labels_list

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        self.transforms = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            normalize])

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.label[index]

        data = pil_loader(img_path)
        data = self.transforms(data)

        label = torch.tensor(label)
        return data, label

    def __len__(self):
        return len(self.data)


def pre(root):
    os.makedirs('image/result/orgin_0')
    os.makedirs('image/result/orgin_1')

    dataset = preData(root)
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = Model("resnet50")
    if torch.cuda.is_available():
        # model.backbone = nn.DataParallel(model.backbone).cuda()
        model.backbone = model.backbone.cuda()

    model.backbone.load_state_dict(torch.load(
        'checkpoints/epoch/40.pth'))
    model.backbone.eval()

    # 统计图片总量
    count = 0

    zero_count = 0
    one_count = 0

    for index, (data, label) in enumerate(dataloader):
        input = Variable(data)
        label = Variable(label)

        if torch.cuda.is_available():
            input = input.cuda()
            label = label.cuda()

        predict = torch.argmax(model.backbone(input), dim=1)

        # 将图片路径和标签保存在excel表中
        res_pre = predict.cpu().numpy().tolist()

        for i in range(0, len(res_pre)):
            if com_label[index*256+i] != res_pre[i]:

                count += 1
                # 样本标为0
                if com_classes[index*256+i].split('/')[2] == '0':
                    zero_count += 1
                    copy(com_classes[index*256+i], os.path.join('image/result/orgin_0',
                                                                com_classes[index*256+i].split('/')[-1]))
                # 样本标为1
                if com_classes[index*256+i].split('/')[2] == '1':
                    one_count += 1
                    copy(com_classes[index*256+i], os.path.join('image/result/orgin_1',
                                                                com_classes[index*256+i].split('/')[-1]))
    print("不匹配图片总数为：", count)
    print("正样本冲突：{}，负样本冲突：{}".format(one_count, zero_count))


if __name__ == "__main__":
    start = datetime.datetime.now()
    # 如果文件夹不存在，新建；存在，删除  
    if not os.path.isdir('image/result'):
        pre("image/imgs")
    else:
        shutil.rmtree('image/result')
        pre("image/imgs")
    end = datetime.datetime.now()
    print(end-start)
