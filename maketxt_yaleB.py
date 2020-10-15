###################  用于生成数据集txt文件   ###########

import os
import numpy as np
from PIL import Image

def read_directory(directory_name):
    global array_of_img  # 声明全局变量
    global class_id
    array_of_img = np.array([])  # this if for store all of the image data
    class_id = np.array([])    #初始化类别标签
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #img is used to store the image data
        if os.path.splitext(filename)[1] == '.pgm':   ### {  检查后缀名  }
            img = Image.open(directory_name + "/" + filename)
            path = directory_name +'/' + filename #just for test
            EYB.write(path+' '+ directory_name[-2:] + '\n')
EYB = open(r'./image/yaleB.txt', 'w')
for filename in os.listdir("./image/EYB"):
    if os.path.splitext(filename)[1] != '.txt':
        read_directory("./image/EYB/" + filename)
EYB = open(r'./image/yaleB.txt', 'r')
train = open(r'./image/EYB/train.txt', 'w')
test = open(r'./image/EYB/test.txt', 'w')
a = 0
for line in EYB:
    if a < 5:
        train.write(line)
    else:
        test.write(line)
    a = a+1
    if a == 10:
        a = 0