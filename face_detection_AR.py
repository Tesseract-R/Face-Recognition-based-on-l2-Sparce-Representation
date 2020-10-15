import cv2
import numpy as np

img = cv2.imread('./image/AR/m-001-12.pgm' , cv2.IMREAD_GRAYSCALE)

face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faces = face_engine.detectMultiScale(img, scaleFactor=1.01, minNeighbors=1)
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
    #cv2.rectangle(img,(x,y,w,h),255,2)
    img = img[y:y+h, x:x+w]
img = cv2.resize(img, (128, 128))
cv2.imshow('img', img)
cv2.waitKey(0)
