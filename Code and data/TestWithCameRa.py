from numpy import asarray
from numpy import expand_dims
from keras.models import load_model
import numpy as np
import pickle
import cv2
import mtcnn
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
#Load MTCCN and FaceNet
face_detector=mtcnn.MTCNN()
MyFaceNet=load_model('./model/facenet_keras.h5')
Normalizer=Normalizer('l2')
print('Load Model Success')
#truy xuất data
myfile=open("data.pkl","rb")
database=pickle.load(myfile)
myfile.close()
#camera
cap=cv2.VideoCapture("http://192.168.1.2:4747/video")
while cap.isOpened():
    _,img=cap.read()
    # img = cv2.imread(img)
    # img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img)
    for res in results:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        confidence = res['confidence']
        if confidence < 0.9:
            continue
        img_face = img[y1:y2, x1:x2]
        # resize ảnh face
        img_face = cv2.resize(img_face, (160, 160))
        # Chuyển về np_arr
        img_face_arr = asarray(img_face)
        img_face_arr = img_face_arr.astype('float32')
        # Tính xác suất
        mean, std = img_face_arr.mean(), img_face_arr.std()
        face = (img_face_arr - mean) / std

        face = expand_dims(face, axis=0)
        # Đưa vào FaceNet để chuyển thành Vector 128
        signature = MyFaceNet.predict(face)
        # Chuẩn hóa Vector
        signature = Normalizer.transform(signature)[0]

        min_dist = 0.2
        indentify = 'Unknow'
        for key, value in database.items():
            dist = cosine(value, signature)
            if dist < min_dist:
                min_dist = dist
                indentify = key
        if indentify=='Unknow':
            cv2.putText(img, indentify, (x1 - 30, y1 - 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.putText(img, indentify + f'__{min_dist:.2f}', (x1 - 30, y1 - 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('res', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()