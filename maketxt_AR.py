###################  用于生成数据集txt文件   ###########


# Note: 1~13 第一周拍摄 14~26 第二周拍摄
# 8~10 sunglass  11~13 scarf

import os
import numpy as np
from PIL import Image
train = open(r'./image/AR/train.txt', 'w')
test = open(r'./image/AR/test.txt', 'w')
AR = open(r'./image/AR.txt', 'w')
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        if os.path.splitext(filename)[1] == '.pgm':   ### {  检查后缀名  }
            tag = filename.split('-')
            index = int(tag[2].split('.')[0])
            if tag[0] == 'w':
                #break
                Person_class = int(tag[1]) + 50
            else:
                Person_class = int(tag[1])
            if index <= 7 or 14 <= index <= 20:
                path = directory_name +'/' + filename #just for test
                train.write(path+' '+ str(Person_class) + '\n')
            elif 11 <= index <= 13 or 24 <= index <= 26:
                path = directory_name + '/' + filename  # just for test
                test.write(path+' '+ str(Person_class) + '\n')


read_directory('./image/AR')
AR = open(r'./image/AR.txt', 'r')

#a = 0
#for line in AR:
#    if a < 7:
#        train.write(line)
#    else:
#        test.write(line)
 #   a = a+1
 #   if a == 14:
  #      a = 0
train.close()
test.close()
AR.close()