import os
import shutil

image_dir = './images/1'
output_dir = './data/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

names = os.listdir(image_dir)

single_size = len(names)//5+1
n = 'a'
for i in range(0, len(names), single_size):
    curr_size = min(single_size, len(names)-i)
    print(curr_size)
    curr_dir = os.path.join(output_dir, n)
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)

    def f(x):
        shutil.copy(
            os.path.join(image_dir, names[x]),
            os.path.join(curr_dir, names[x])
        )
    list(map(f, range(i, i+curr_size)))
    n = chr(ord(n)+1)
