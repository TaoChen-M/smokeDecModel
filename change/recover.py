'''
-*- encoding: UTF-8 -*-
Description      :如若修改出错，根据image_index.txt还原所有的图片路径
'''

import os
import shutil


def move(src_path, dst_path):
    print(src_path, dst_path, end=' ')
    if os.path.exists(src_path):
        print(True)
        shutil.move(src_path, dst_path)
    else:
        print(False)


if __name__ == '__main__':
    with open('image_index.txt', 'r') as f:
        dst_paths = f.read().strip().split('\n')

    src_paths = [p[:-5] + str(1 - int(p[-5])) + p[-4:] for p in dst_paths]
    list(map(move, src_paths, dst_paths))
