from os import path
import torch
import torchvision.models as models
from net.model import Model
from PIL import Image
from test import preData
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from shutil import copy
from pathlib import Path
from torch.utils import data
from torchvision import transforms as T
# load image
path="image/mat"

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

if not os.path.isdir('image/result'):
    os.makedirs('image/result/0')
    os.makedirs('image/result/1')

dataset = preData(path)
dataloader = DataLoader(dataset, batch_size=16,
                        shuffle=False, num_workers=4, pin_memory=True)

model=models.resnet50()
model.fc = torch.nn.Linear(2048, 2)
torch_model = torch.load("checkpoints/epoch/19.pth") # pytorch模型加载
model.load_state_dict(torch_model) 

# #set the model to inference mode
model.eval()
# 统计两个文件夹图片数量
zero_count = 0
one_count = 0
for index, data in enumerate(dataloader):
    input = Variable(data)

    predict = torch.argmax(model(input), dim=1)

    res_pre = predict.numpy().tolist()
    for i in range(0, len(res_pre)):
        if res_pre[i] == 0:
            zero_count += 1
            copy(com_classes[index*16+i], os.path.join('image/result/0',
                    com_classes[index*16+i].split('/')[-1]))
        else:
            one_count += 1
            copy(com_classes[index*16+i], os.path.join('image/result/1',
                    com_classes[index*16+i].split('/')[-1]))

print("预测为负样本图片：", zero_count)
print("预测为正样本图片：", one_count)
