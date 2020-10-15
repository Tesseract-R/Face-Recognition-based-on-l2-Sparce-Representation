###################  用于生成数据集txt文件   ###########

import os
import numpy as np
from PIL import Image
path = open('./image/lfw/AllNames.txt', 'r')
lfw_select = open(r'./image/lfw_selected_cropped.txt', 'w')
train = open(r'./image/lfw_cropped/train.txt', 'w')
test = open(r'./image/lfw_cropped/test.txt', 'w')
class_total = 0
for line in path:
    line = line.rstrip()  #删除末尾字符
    words = line.split()  #字符串切片
    path_in = './image/lfw/'+words[0]
    if len(os.listdir(path_in)) > 50:
        print(words[0],len(os.listdir(path_in)))
        class_total = class_total + 1
        a = 0
        for filename in os.listdir(path_in):
            lfw_select.write('./image/lfw_cropped/' + filename + ' ' + words[0]+'\n')
            lfw_select.flush()
            a = a+1
            if a == 50:
                break
lfw_select.close()
print('class = ', class_total)

lfw = open(r'./image/lfw_selected_cropped.txt', 'r')

a = 0
for line in lfw:
    if a < 5:
        train.write(line)
        train.flush()
    else:
        test.write(line)
        test.flush()
    a = a+1
    if a == 10:
        a = 0
train.close()
test.close()
lfw.close()