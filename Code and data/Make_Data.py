import os
from os import listdir
from numpy import asarray
from numpy import expand_dims
from keras.models import load_model
import numpy as np
import pickle
import cv2
import mtcnn
from sklearn.preprocessing import Normalizer
#Load MTCCN and FaceNet
face_detector=mtcnn.MTCNN()
MyFaceNet=load_model('./model/facenet_keras.h5')
print('Load Model Success')
Normalizer=Normalizer('l2')
#Khai bao thư mục chứa data
folder='dataset/train/'
database={}
conf_t=0.9
#Xử lý ảnh input
for folder_sub in listdir(folder):
    print(folder_sub)
    path=folder+folder_sub+'/'
    data = []
    for file_name in listdir(path):

        print(file_name)
        image=cv2.imread(path+os.sep+file_name)
        # cv2.imshow('cap',image)
        # cv2.waitKey(0)
        # img_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results=face_detector.detect_faces(image)
        for res in results:
            x1, y1, width, height = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            confidence = res['confidence']
            #Nếu độ tin bé quá thì bỏ qua
            if confidence < conf_t:
                continue
            # img_face=img_rgb[y1:y2,x1:x2]
            img_face = image[y1:y2, x1:x2]
            #Resize Image for Input FaceNet
            img_face=cv2.resize(img_face,(160,160))
            #Chuyển về np_arr
            img_face_arr=asarray(img_face)
            img_face_arr=img_face_arr.astype('float32')
            #Tính xác suất
            mean,std = img_face_arr.mean(),img_face_arr.std()
            face=(img_face_arr-mean)/std
            face=expand_dims(face,axis=0)
            #Đưa vào mạng FaceNet để chuyển thành Vecto 128
            signature=MyFaceNet.predict(face)
            #Add vào database
            data.append(signature)
    #Thêm
    if data:
        signature=np.sum(data,axis=0)
        signature=Normalizer.transform(signature)[0]
        database[folder_sub]=signature

myfile=open("data.pkl","wb")
pickle.dump(database,myfile)
myfile.close()
print(database)






