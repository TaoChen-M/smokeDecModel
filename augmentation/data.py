from PIL import Image
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import random

from torchvision.transforms.transforms import CenterCrop, Resize


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img

        # # 随机拉伸处理
        # scale1 = random.uniform(0.7, 1.5)
        # scale2 = random.uniform(0.7, 1.5)

        # width = (int)(img.size[0]*scale1)
        # height = (int)(img.size[1]*scale2)

        # return img.resize((width, height), Image.ANTIALIAS)


def transfer(root):
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

    # 随机打乱数据
    np.random.seed(0)
    np.random.shuffle(classes_list)
    np.random.seed(0)
    np.random.shuffle(labels_list)

    return classes_list, labels_list


########k折划分############
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = [], []
    for j in range(k):
        # slice(start,end,step)切片函数
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # idx 为每组 valid
        X_part, y_part = X[idx], y[idx]
        if j == i:  # 第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train.extend(X_part)
            y_train.extend(y_part)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid


class smokeData(data.Dataset):
    def __init__(self, data, label, train):
        # super().__init__()
        self.data = data
        self.label = label

        # 现在使用的高斯噪声是ImageNet的，考虑自己设置高斯噪声
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        if train:
            self.transforms = T.Compose([
                # resize imgs to 224*224
                T.Resize((64,64)),
                T.CenterCrop((64,64)),
                # T.Resize((224,224)),
                # T.CenterCrop((224,224)),
                # 随机水平翻转
                T.RandomHorizontalFlip(0.5),
                # 随机垂直翻转
                T.RandomVerticalFlip(0.5),
                # 随机旋转正负90度
                # T.RandomRotation([90, 270]),
                # 随机改变亮度、对比度
                # T.ColorJitter(brightness=0.5, contrast=0.5),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((64,64)),
                T.CenterCrop((64,64)),
                # T.Resize((224,224)),
                # T.CenterCrop((224,224)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.label[index]

        data = pil_loader(img_path)
        data = self.transforms(data)

        label = torch.tensor(label)
        return data, label

    def __len__(self):
        return len(self.data)
