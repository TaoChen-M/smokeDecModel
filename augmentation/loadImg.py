'''
-*- encoding: UTF-8 -*-
Description      :将图片从patches中复制到imgs中，作为训练数据，同时创建原图片路径和图片索引之间的映射
'''
import os
import time
from shutil import copy
import datetime
import shutil


def run(root, txt, imgs):
    start = datetime.datetime.now()
    print("start to load images from {} to {}".format(root, imgs))
    # 删除image_index.txt  imgs文件夹
    if os.path.exists(txt):
        os.remove(txt)
    if os.path.isdir(imgs):
        shutil.rmtree(imgs)

    # 创建目标文件夹
    zero_path = os.path.join(imgs, '0')
    one_path = os.path.join(imgs, '1')

    os.makedirs(zero_path)
    os.makedirs(one_path)

    # 图片索引
    zero_index = 0
    one_index = 0
    img_index = 0

    for home, sub_dirs, files in os.walk(root):
        for file in files:
            img_index += 1
            # 将每次读取的图片写入txt文件
            with open(txt, 'a') as f:
                f.writelines(str(home)+'/'+str(file))
                f.write('\n')

            patch_index, label = file.split('.')[0].split('_')
            # 图片名为按顺序的索引
            if label == '0':
                zero_index += 1
                copy(os.path.join(home, file),
                     os.path.join(zero_path, str(img_index)+'.'+file.split('.')[-1]))
            else:
                one_index += 1
                copy(os.path.join(home, file),
                     os.path.join(one_path, str(img_index)+'.'+file.split('.')[-1]))

    end = datetime.datetime.now()

    print("总样本数量：{}，正样本数量：{}，负样本数量：{}".format(
        img_index, one_index, zero_index))

    print("time waste to loadimg:", end-start)


if __name__ == '__main__':
    run('/datasets/patches_new/', './change/image_index.txt', 'image/imgs')
