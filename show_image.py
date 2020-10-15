#############  显示图片  ###########

import cv2
import os
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing


#读取单张图片
from PIL import Image

def read_directory(directory_name):
    global array_of_img  # 声明全局变量
    global class_id
    array_of_img = np.array([])  # this if for store all of the image data
    class_id = np.array([])    #初始化类别标签
    hstack_1 = []
    hstack_2 = []
    i = 0
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data
        if os.path.splitext(filename)[1] == '.pgm':   ### {  检查后缀名  }
            img = cv2.imread(directory_name + "/" + filename, cv2.IMREAD_GRAYSCALE)
            i = i +1
            if i <= 50:
                if hstack_1 == []:
                    hstack_1 = img
                else:
                    hstack_1 = np.hstack((hstack_1, img))
            else:
                if hstack_2 == []:
                    hstack_2 = img
                else:
                    hstack_2 = np.hstack((hstack_2, img))
            if i == 100:
                img = np.vstack((hstack_1, hstack_2))
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break


read_directory("./image/AR")
#for filename in os.listdir("./image/ORL_Faces"):
#   if os.path.splitext(filename)[1] != '.txt':
#       read_directory("./image/ORL_Faces" + "/" + filename)
