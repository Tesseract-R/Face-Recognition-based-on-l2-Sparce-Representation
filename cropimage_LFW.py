import cv2
import os
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from PIL import Image

path = './image/lfw-deepfunneled/'
name = open('./image/lfw/AllNames.txt', 'r')
for line in name:
    line = line.rstrip()  #删除末尾字符
    words = line.split()  #字符串切片
    for filename in os.listdir(path + words[0]):
        img = cv2.imread(path + words[0] + '/' + filename, cv2.IMREAD_GRAYSCALE)
        face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        faces = face_engine.detectMultiScale(img,scaleFactor=1.1, minNeighbors=1)
        print(filename)
        if type(faces) is tuple:
            continue
        else:
            if faces.size > 4:
                size = faces[:, 3]
                pos = np.argmax(size)
                faces = faces[pos]
                (x, y, w, h) = faces
            else:
                (x, y, w, h) = faces[0]
            img = img[y:y + h, x:x + w]
        img = cv2.resize(img, (128, 128))
        cv2.imwrite(r'C:\Users\zrc5\PycharmProjects\untitled\image\lfw_cropped'+'/'+filename, img)