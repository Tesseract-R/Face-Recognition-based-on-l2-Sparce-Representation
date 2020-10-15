import cv2
import numpy as np

nametag = 'George_W_Bush_0284.jpg'
name = nametag.split('.')[0]
print(len(nametag.split('_')))
name = nametag.split('_')[0]+'_'+nametag.split('_')[1]+'_'+nametag.split('_')[2]
img = cv2.imread('./image/lfw-deepfunneled/' + name + '/' + nametag, cv2.IMREAD_GRAYSCALE)
img = cv2.imread('./image/lena.bmp')
face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faces = face_engine.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1)
print(faces)
if type(faces) is tuple:
    print('1')
else:
    if faces.size > 4:
        size = faces[:,3]
        pos = np.argmax(size)
        faces = faces[pos]
        (x, y, w, h) = faces
    else:
        (x,y,w,h) = faces[0]
    cv2.rectangle(img,(x,y,w,h),255,2)
    #img = img[y:y+h, x:x+w]
#img = cv2.resize(img, (128, 128))
cv2.imshow('img', img)
cv2.waitKey(0)
#print(img.shape)
#cv2.imwrite('./image/lfw_cropped/'+ nametag, img)