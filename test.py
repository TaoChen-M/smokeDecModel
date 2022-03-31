import torch
from pathlib import Path
from torch.utils import data
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.transforms import CenterCrop
from xlwt.Workbook import Workbook
from net.model import Model
from torch.autograd import Variable
from torch import nn
from xlutils.copy import copy
import datetime
import os
from shutil import copy
import numpy as np
import random


com_classes = []


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class preData(data.Dataset):
    def __init__(self, root):
        dir_path = Path(root)
        images = []

        for img in dir_path.glob('*.png'):
            images.append(str(img))

        global com_classes
        com_classes = images
        self.data = images

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        self.transforms = T.Compose([
            T.Resize((224,224)),
            T.CenterCrop((224,224)),
            T.ToTensor(),
            normalize])

    def __getitem__(self, index):
        img_path = self.data[index]

        data = pil_loader(img_path)
        data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.data)


def pre(root):
    if not os.path.isdir('image/result'):
        os.makedirs('image/result/0')
        os.makedirs('image/result/1')

    dataset = preData(root)
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = Model("resnet50")
    if torch.cuda.is_available():
        # model.backbone = nn.DataParallel(model.backbone).cuda()
        model.backbone = model.backbone.cuda()

    model.backbone.load_state_dict(torch.load(
        'checkpoints/epoch/91.pth'))
    model.backbone.eval()

    # 统计两个文件夹图片数量
    zero_count = 0
    one_count = 0
    for index, data in enumerate(dataloader):
        input = Variable(data)

        if torch.cuda.is_available():
            input = input.cuda()

        predict = torch.argmax(model.backbone(input), dim=1)

        res_pre = predict.cpu().numpy().tolist()
        for i in range(0, len(res_pre)):
            if res_pre[i] == 0:
                zero_count += 1
                copy(com_classes[index*256+i], os.path.join('image/result/0',
                     com_classes[index*256+i].split('/')[-1]))
            else:
                one_count += 1
                copy(com_classes[index*256+i], os.path.join('image/result/1',
                     com_classes[index*256+i].split('/')[-1]))

    print("预测为负样本图片：", zero_count)
    print("预测为正样本图片：", one_count)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    start = datetime.datetime.now()
    pre("./patches")
    end = datetime.datetime.now()
    print(end-start)


