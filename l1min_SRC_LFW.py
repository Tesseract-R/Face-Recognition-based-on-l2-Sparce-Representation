###################  基于稀疏表示的人脸识别   ###########
### 所用测试集：LFW (开集识别)
### 采用下采样降低维数，采样倍率：8   维度：1024
### 稀疏表示求解方法：OMP

import cv2
import os
import numpy as np
from sklearn import preprocessing
from PIL import Image
import time

start =time.clock()


####通过train.txt test.txt作为索引导入数据 ####

path = open('./image/lfw_cropped/train.txt', 'r')
img_train = np.array([])
class_id_train = np.array([])
tag_saver = []
for line in path:
    line = line.rstrip()  #删除末尾字符
    words = line.split()  #字符串切片
    img = Image.open(words[0])
    img = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(np.array(img))))  # 降采样 倍数为8
    img_ravel = np.reshape(img, (-1, 1))  # 降维至列向量
    img_ravel = np.array(img_ravel)
    if img_train.size > 0:
        img_train = np.column_stack((img_train, img_ravel))  # 放在一个矩阵中
    else:
        img_train = img_ravel
    tag = (words[1])
    if len(tag_saver) == 0:
        tag_saver = [tag]
    if tag_saver[-1] != tag:
        tag_saver.append(tag)
    if class_id_train.size > 0:
        class_id_train = np.hstack((class_id_train, tag))  # 放在一个矩阵中
    else:
        class_id_train = np.array([tag])
path = open('./image/lfw_cropped/test.txt', 'r')
img_test = np.array([])
class_id_test = np.array([])
for line in path:
    line = line.rstrip()  #删除末尾字符
    words = line.split()  #字符串切片
    img = Image.open(words[0])
    img = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(np.array(img))))  # 降采样 倍数为8
    img_ravel = np.reshape(img, (-1, 1))  # 降维至列向量
    img_ravel = np.array(img_ravel)
    if img_test.size > 0:
        img_test = np.column_stack((img_test, img_ravel))  # 放在一个矩阵中
    else:
        img_test = img_ravel
    tag = (words[1])
    if class_id_test.size > 0:
        class_id_test = np.hstack((class_id_test, tag))  # 放在一个矩阵中
    else:
        class_id_test = np.array([tag])

class_id_train = np.array(class_id_train)
img_train_norm = preprocessing.normalize(img_train.T, norm='l2')   #l2 norm 归一化
img_test_norm = preprocessing.normalize(img_test.T, norm='l2')
img_train = img_train.T
img_train_norm = img_train_norm.T

#####################    OMP求解稀疏表示  ###############
def cs_omp(y, D):
    L = y.shape[0]
    phi = np.array([[]])   #选出原子组成的矩阵
    residual = y  #初始化残差
    index = np.zeros((L), dtype=int)  #选出原子的索引集
    result = np.zeros((img_train_norm.shape[1]), dtype=float)  #待求的系数
    for i in range(L):
        index[i] = -1
    for j in range(L):  #迭代次数
        product = np.fabs(np.dot(D.T, residual))
        pos = np.argmax(product)  #最大投影系数对应的位置
        index[j] = pos
        if phi.size > 0:
            newstack = np.mat(D[:, pos]).T
            phi = np.hstack((phi, newstack))
        else:
            phi = np.mat(D[:, pos]).T
        a = np.dot(np.linalg.pinv(phi), y)
        residual = y - np.dot(phi, a)
        if np.linalg.norm(residual) < 0.01:
            break
    for i in range(a.size):
        result[index[i]] = a[i]
    result = np.mat(result).T   #稀疏系数
    return result

#开始测试
sucess_time = 0
for test_time in range(img_test_norm.shape[0]):
    class_of_test = class_id_test[test_time]   #测试图像所属的类别
    print('测试类别：', class_of_test)
    y = img_test_norm[test_time]   #选取测试图像
    y = np.mat(y).T
    coefficient_x = cs_omp(y, img_train_norm)
    residual = np.ones(len(tag_saver), dtype=float)  #初始化重建误差
    class_id_train = np.array(class_id_train)
    for i in range(len(tag_saver)):
        sigma_x = np.where(class_id_train == tag_saver[i])
        A_i = np.mat(img_train_norm[:, np.mat(sigma_x)])
        x_i = np.reshape(coefficient_x[np.mat(sigma_x)], (-1, 1))
        residual[i] = np.linalg.norm(y - np.dot(A_i, x_i))
    pos = np.argmin(residual)
    class_of_result = tag_saver[pos]
    print('分类结果', class_of_result)
    if class_of_result == class_of_test:
        sucess_time = sucess_time + 1
    print('第', test_time + 1, '次试验，准确率：', sucess_time/(test_time+1)*100, '%')
    print('****************************************')
end = time.clock()
print('Running time: %s Seconds'%(end-start))