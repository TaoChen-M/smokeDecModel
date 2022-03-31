'''
-*- encoding: UTF-8 -*-
Description      :1、需要修改后缀的图片分别放置在0和1的文件夹中，0文件夹为负样本图片，1文件夹为正样本图片
                  2、根据图片名称和image_index.txt映射修改，例如 1.png，则image_index.txt第一行即为其路径
                  
'''

import os
result = []
f = open('image_index.txt', 'r')
lines = f.readlines()

for line in lines:
    result.append(line)

print("总图片数量为：", len(result))

path = 'old'

zero = []
one = []

for item in os.listdir(path):
    for name in os.listdir(os.path.join(path, item)):
        if item == '0':
            zero.append(int(name.split('.')[0]))
        if item == '1':
            one.append(int(name.split('.')[0]))


print("length of zero", len(zero))
print("length of one", len(one))

zero.sort()
one.sort()
for i in zero:
    path = result[i-1]
    if path.split('/')[-1].split('_')[1].split('.')[0] == '0':
        continue
    if path.split('/')[-1].split('_')[1].split('.')[0] == '1':
        newname = path[:-6]+str('0.png')
        os.rename(path[:-1], newname)

for i in one:
    path = result[i-1]
    if path.split('/')[-1].split('_')[1].split('.')[0] == '1':
        continue
    if path.split('/')[-1].split('_')[1].split('.')[0] == '0':
        newname = path[:-6]+str('1.png')
        os.rename(path[:-1], newname)

print("change sucess")
