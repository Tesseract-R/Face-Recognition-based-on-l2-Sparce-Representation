###################  基于稀疏表示的人脸识别   ###########
### 所用测试集：Extended yale B
### 采用下采样降低维数，采样倍率：8
### 稀疏表示求解方法：OMP

import cv2
import numpy as np
from sklearn import preprocessing
from PIL import Image
import matplotlib.pyplot as plt
import time

start =time.process_time()


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
    img = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(np.array(img)))) # 降采样 倍数为8
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
img_train_norm = img_train_norm.T

class_id_train = np.array(class_id_train)
img_train_norm = preprocessing.normalize(img_train.T, norm='l2')   #l2 norm 归一化
img_test_norm = preprocessing.normalize(img_test.T, norm='l2')
img_train = img_train.T
img_train_norm = img_train_norm.T
#print(img_train_norm.shape)

#####################    OMP求解稀疏表示  ###############
def cs_omp(y, D):
    L = y.shape[0]
    k = 0
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
        k = k+1
        if phi.size > 0:
            newstack = np.mat(D[:, pos]).T
            phi = np.hstack((phi, newstack))
        else:
            phi = np.mat(D[:, pos]).T
        a = np.dot(np.linalg.pinv(phi), y)
        residual = y - np.dot(phi, a)
        if k >= 70:    #k用于保证稀疏度
            break
    for i in range(a.size):
        result[index[i]] = a[i]
    result = np.mat(result).T   #稀疏系数
    return result

#开始测试
sucess_time = 0
gamma = 0.001
L = len(tag_saver)
M = np.array([])
for i in range(L):
    sigma_x = np.where(class_id_train == tag_saver[i])
    A_i = np.mat(img_train_norm[:, np.mat(sigma_x)])
    M_i = np.dot(A_i.T, A_i)
    if M.size == 0:
        M = M_i
    else:
        M_up = np.concatenate((M, np.zeros((M.shape[0], M_i.shape[1]))), axis=1)
        M_down = np.concatenate((np.zeros((M_i.shape[0], M.shape[1])), M_i), axis=1)
        M = np.concatenate((M_up, M_down))
x_up = np.linalg.inv((1 + 2 * gamma) * np.dot(img_train_norm.T, img_train_norm) + 2 * gamma * L * M)
for test_time in range(img_test_norm.shape[0]):
    randIndex = test_time
    class_of_test = class_id_test[randIndex]   #测试图像所属的类别
    print('测试类别：', class_of_test)
    y = img_test_norm[randIndex]   #选取测试图像
    y = np.mat(y).T
    x_down = cs_omp(y, img_train_norm)
    x_up_y = np.dot(np.dot(x_up, img_train_norm.T), y)
    coefficient_x = x_up_y/np.linalg.norm(x_up_y) + x_down/np.linalg.norm(x_down)
    #for i in range(1207):
    #    plt.plot([i, i],[0, x_down[i]])
    #plt.show()
    residual = np.ones(L, dtype=float)  #初始化重建误差
    class_id_train = np.array(class_id_train)
    for i in range(L):
        sigma_x = np.where(class_id_train == tag_saver[i])
        A_i = np.mat(img_train_norm[:, np.mat(sigma_x)])
        x_i = np.reshape(coefficient_x[np.mat(sigma_x)], (-1, 1))
        residual[i] = np.linalg.norm(y - np.dot(A_i, x_i))/np.linalg.norm(x_i)
    pos = np.argmin(residual)
    class_of_result = tag_saver[pos]
    print('分类结果', class_of_result)
    if class_of_result == class_of_test:
        sucess_time = sucess_time + 1
    print('第', test_time + 1, '次试验，准确率：', sucess_time/(test_time+1)*100, '%')
    print('****************************************')

end = time.clock()
print('Running time: %s Seconds'%(end-start))