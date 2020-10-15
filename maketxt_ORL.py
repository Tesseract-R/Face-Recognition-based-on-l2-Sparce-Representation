###################  用于生成数据集txt文件   ###########

import os
import numpy as np
from PIL import Image

def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #img is used to store the image data
        if os.path.splitext(filename)[1] == '.pgm':   ### {  检查后缀名  }
            path = directory_name +'/' + filename #just for test
            ORL.write(path+' '+ directory_name[-2:] + '\n')
ORL = open(r'./image/ORL.txt', 'w')
for filename in os.listdir("./image/ORL_Faces"):
    if os.path.splitext(filename)[1] != '.txt':
        read_directory("./image/ORL_Faces/" + filename)
ORL = open(r'./image/ORL.txt', 'r')
train = open(r'./image/ORL_Faces/train.txt', 'w')
test = open(r'./image/ORL_Faces/test.txt', 'w')
a = 0
for line in ORL:
    if a < 7:
        train.write(line)
    else:
        test.write(line)
    a = a+1
    if a == 10:
        a = 0
train.close()
test.close()
ORL.close()