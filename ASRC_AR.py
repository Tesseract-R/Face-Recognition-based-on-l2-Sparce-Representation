###################  基于自适应稀疏表示的人脸识别   ###########
### 所用测试集：AR
### 采用下采样降低维数，采样倍率：8
### 稀疏表示求解方法：ADM

import cv2
import os
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from PIL import Image
from scipy.optimize import minimize
import time

start =time.clock()

####通过train.txt test.txt作为索引导入数据 ####

path = open('./image/AR/train.txt', 'r')
img_train = np.array([])
class_id_train = np.array([])
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
    tag = int(words[1])
    if class_id_train.size > 0:
        class_id_train = np.hstack((class_id_train, tag))  # 放在一个矩阵中
    else:
        class_id_train = np.array([tag])
path = open('./image/AR/test.txt', 'r')
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
    tag = int(words[1])
    if class_id_test.size > 0:
        class_id_test = np.hstack((class_id_test, tag))  # 放在一个矩阵中
    else:
        class_id_test = np.array([tag])

class_id_train = np.array(class_id_train)
img_train_norm = preprocessing.normalize(img_train.T, norm='l2')   #l2 norm 归一化
img_test_norm = preprocessing.normalize(img_test.T, norm='l2')
img_train = img_train.T
img_train_norm = img_train_norm.T
#print(img_train.shape)

##### 奇异值阈值法 #####
def cs_svt(mat, tau):
    u, s, v = np.linalg.svd(mat, full_matrices=0)
    vec = s - tau
    vec[vec < 0] = 0
    return np.matmul(np.matmul(u, np.diag(vec)), v)

##### 软阈值法 #####
def soft_threshold(b, lamb):
    soft_thresh = np.dot(np.sign(b), max((abs(b.all()) - lamb/2, 0)))
    return soft_thresh

#####################    ADM(Alternating Direction Method)求解稀疏表示  ###############
def cs_ADM(y, D):
    e = y
    alpha = np.zeros((img_train_norm.shape[1], 1))
    diag_a = np.diag(np.array(alpha)[:, 0])
    y1 = e
    Y2 = np.zeros((img_train_norm.shape[0], img_train_norm.shape[1]))
    J = np.dot(D, diag_a)
    rou = 0.6
    mu = 0.6
    mu_max = 0.9
    lamb = 0.8
    loop_time = 0
    while True:
        J = cs_svt(np.dot(D, diag_a) - 1/mu*Y2, lamb/mu)
        A = np.dot(D.T, D) + np.diag(np.diag(np.dot(D.T, D)))
        A = np.linalg.inv(A)
        alpha = np.dot(np.dot(A, D.T), (y1/mu + y - e)) + np.dot(A, np.mat(np.diag(np.dot(D.T, (Y2/mu + J)))).T)
        diag_a = np.diag(np.array(alpha)[:, 0])
        e = soft_threshold(y - np.dot(D, alpha) + 1/mu*y1, 1/mu)
        y1 = y1 + mu*(y - np.dot(D, alpha) - e)
        Y2 = Y2 + mu*(J - np.dot(D, diag_a))
        mu = np.min((rou*mu, mu_max))
        residual_1 = y - np.dot(D, alpha) - e
        residual_2 = J - np.dot(D, diag_a)
        loop_time += 1
        if loop_time > 20:
            break
        if np.linalg.norm(residual_1, ord=np.inf) <= 0.5 and np.linalg.norm(residual_2, ord=np.inf) <= 1:
            break
    result = alpha   #稀疏系数
    return result

#开始测试
sucess_time = 0
L = 50
for test_time in range(img_test_norm.shape[0]):
    randIndex = test_time
    class_of_test = class_id_test[randIndex]   #测试图像所属的类别
    print('测试类别：', class_of_test)
    y = img_test_norm[randIndex]   #选取测试图像
    y = np.mat(y).T
    coefficient_x = cs_ADM(y, img_train_norm)
    residual = np.ones(L, dtype=float)  #初始化重建误差
    class_id_train = np.array(class_id_train)
    for i in range(L):
        sigma_x = np.where(class_id_train == i+1)
        A_i = np.mat(img_train_norm[:, np.mat(sigma_x)])
        x_i = np.reshape(coefficient_x[np.mat(sigma_x)], (-1, 1))
        residual[i] = np.linalg.norm(y - np.dot(A_i, x_i))
    pos = np.argmin(residual)
    print('分类结果', pos+1)
    if pos+1 == class_of_test:
        sucess_time = sucess_time + 1
    print('第', test_time + 1, '次试验，准确率：', sucess_time/(test_time+1)*100, '%')
    print('****************************************')
end = time.clock()
print('Running time: %s Seconds'%(end-start))