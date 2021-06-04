import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import os
import pickle
from scipy import signal
from datetime import datetime


def uniform(img,kernel_size = 3):
  kernel = (1/kernel_size**2)*np.ones([kernel_size,kernel_size])
  # 'boundary' tells the function to implement the convolution in a symmetrical 
  # way. 'same' produces an image with the same size as the input image
  filtered = signal.convolve2d(img, kernel, boundary='symm', mode='same')
  return filtered

def mid_filter(img, size):
  m, n = img.shape
  filtered = np.zeros([m,n])
  # Let's pad the image with extra zeros
  gray_image_v2 = np.zeros([m+2, n+2])
  gray_image_v2[1:m+1,1:n+1] = img
  step_size = int((size-1)/2)
  for i in range(m):
    for j in range(n):
      filtered[i,j] = 1/2.0*(np.max(gray_image_v2[i-step_size +1 :i+step_size +1, j-step_size +1 : j+step_size +1])+ np.min(gray_image_v2[i-step_size +1 :i+step_size +1,
                                                  j-step_size +1 : j+step_size +1]))
  return filtered

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f: # r+ es para leer y escribir
        myDataList = f.readlines()          # lista de asistencia
        nameList = []                       # Lista para poner los nombres
        for line in myDataList:
            entry = line.split(',')         # Se separa cada dato con la coma
            nameList.append(entry[0])       # Hace append de los nombres
        if name not in nameList:                    # Solo marca la asistencia si la persona no marcó antes
            now = datetime.now()                    # Tiempo actual
            dtString = now.strftime('%H:%M:%S')     # Formato del tiempo hora:minuto:segundo
            f.writelines(f'\n{name}, {dtString}')   # Escribe el nombre y la hora
#import models

face_cascade=cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

#load previously trained model
filename='trained_model.sav'
classifier=pickle.load(open(filename,'rb'))

#open camara
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    img=frame
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, 1.1,4)
    if len(face)==0:
        cv2.putText(img,'Face not detected',(10,450),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255))
        continue
    elif(face[0][3]>=128):
        cv2.putText(img,'Face detected, press Q to exit',(10,450),cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0),thickness=4)
        if cv2.waitKey(1) & 0xFF==ord('q'):
          break
    else:
        cv2.putText(img,'Come closer',(10,450),cv2.FONT_HERSHEY_PLAIN, 2,(0.0,255),thickness=4)
        continue
    cv2.imshow('vid',img)
cap.release()
cv2.destroyAllWindows()

#preprocesing
x,y,w,h=face[0]
gray2=gray[y:y+h,x:x+w]
gray3=cv2.resize(gray2,(128,128))
gray4=uniform(gray3)
gray5=mid_filter(gray4,3)
pca=PCA(n_components=128)
pca_img=pca.fit_transform(gray5)
final_img=pca_img.reshape((-1))

#image clasifier
prediction_Prob=classifier.predict_proba(final_img.reshape(1,-1))
prediction=classifier.predict(final_img.reshape(1,-1))
print(prediction[0])
print(prediction_Prob)
print('Coincidence of:{0}%'.format(float(np.amax(prediction_Prob[0])*100)))

#mark attendance
if np.amax(prediction_Prob[0])<0.80:
  print('Not registered')
  markAttendance('Not Registered')
  cv2.imwrite('unregistered/UnRegistered_person_{0}.jpg'.format(datetime.now().strftime('%H:%M:%S'))) #added funcion for getting images from a Not Registered Person
else:
  markAttendance(prediction[0]) #save attendance in CSV file
