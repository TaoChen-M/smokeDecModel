'''
-*- encoding: UTF-8 -*-
Description      :用来核查需要修改的图片路径能够服务器上的对应
Time             :2021/08/05 16:14:22
Author           :chentao
Version          :1.0
'''

import os
import cv2


def check(src_path, dst_path):
    a, b = cv2.imread(src_path), cv2.imread(dst_path)
    if (a != b).any():
        print(a, b)


if __name__ == '__main__':
    with open('image_index.txt', 'r') as f:
        index_to_path = f.read().strip().split('\n')
    for label in '01':
        filenames = os.listdir(os.path.join('.', label))
        filenames.sort()
        file_paths = [os.path.join('.', label, name) for name in filenames]

        indexes = [int(name.split('.')[0]) - 1 for name in filenames]
        src_paths = [index_to_path[i] for i in indexes]
        for file_path, src_path in zip(file_paths, src_paths):
            check(file_path, src_path)
