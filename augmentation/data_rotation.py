from pathlib import Path
import cv2
import numpy as np
import math
import shutil
import os
import datetime
from .loadImg import run


def rotation(root, txt, imgs, rotationpath):
    run(root, txt, imgs)
    start = datetime.datetime.now()
    print('start to rotate all images from {} to {}'.format(imgs, rotationpath))
    if os.path.isdir(rotationpath):
        shutil.rmtree(rotationpath)

    target_path = Path(rotationpath)
    target_path.mkdir()

    dir_path = Path(imgs)
    classes = []

    for category in dir_path.iterdir():
        if category.is_dir():
            classes.append(category.name)

    classes.sort()

    for index, name in enumerate(classes):
        class_path = dir_path / name
        target_class_path = target_path / name

        if not target_class_path.exists():
            target_class_path.mkdir()
        if not class_path.is_dir():
            continue

        for file in class_path.iterdir():
            file = str(file)
            # print(file)

            img_name = file[file.rfind('/')+1:]

            im = cv2.imread(file)

            points = np.where((im[:, :, 0] == 0) & (
                im[:, :, 1] == 0) & (im[:, :, 2] == 255))

            if len(points[0]) == 0 or len(points[1]) == 0:
                continue

            point0 = [float(points[1][0]), float(points[0][0])]
            point1 = [float(points[1][-1]), float(points[0][-1])]

            center = [(point0[0]+point1[0])/2, (point0[1]+point1[1])/2]

            if point1[0] == point0[0]:
                if point1[1] > point0[1]:
                    angle = 90.0
                else:
                    angle = -90.0
            else:
                angle = (point1[1]-point0[1])/(point1[0]-point0[0])
                angle = math.degrees(math.atan(angle))

            if angle > 90 and angle < 180:
                angle = -(180-angle)

            if angle > 180 and angle < 270:
                angle = angle-180

            if angle > 270:
                angle = -(360-angle)

            M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1)
            dst = cv2.warpAffine(im, M, (60, 60), borderValue=(255, 255, 255))
            M = np.float32([[1, 0, 30-center[0]], [0, 1, 30-center[1]]])
            dst = cv2.warpAffine(dst, M, (60, 60), borderValue=(255, 255, 255))

            target_file = str(target_class_path / img_name)
            cv2.imwrite(target_file, dst)

    end = datetime.datetime.now()
    print("time waste to rotation:", end-start)


if __name__ == '__main__':
    rotation('/home/dataset/patches_new', './change/image_index.txt',
             'image/imgs', 'image/rotation')
