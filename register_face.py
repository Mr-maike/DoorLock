import cv2
import sqlite3
import numpy as np
import os

# Conexão com o banco de dados
conn = sqlite3.connect('usersdatabase.db')
c = conn.cursor()

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

# Carrega os arquivos Haar Cascade
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

cap = cv2.VideoCapture(1)
u_name = input("Informe o seu nome: ")
c.execute('INSERT INTO users (name) VALUES (?)', (u_name,))
uid = c.lastrowid
sampleNum = 0

while True:
  _,img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor= 1.1,
    minNeighbors= 5,
    minSize=(30, 30)
)

  for (x,y,w,h) in faces:

    region = img[y:y + h, x:x + w]
    regionGrayEye = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    DetectedEye = eye_cascade.detectMultiScale(regionGrayEye)

    for (ex, ey, eh, ew) in DetectedEye:

      cv2.rectangle(region, (ex, ey), (ex + ew, ey + eh), (255,0,0), 2)

      if cv2.waitKey(1) & 0xFF== ord('q'):
        sampleNum = sampleNum+1
        print('[SISTEMA] Foto ' + str(sampleNum) + ' Capturada com sucesso!')
        faceImg = cv2.resize(gray[y:y + h, x:x + w], (220,220))
        cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg", faceImg)

    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

  cv2.imshow('Registro de Imagens',img)
  cv2.waitKey(1)

  if sampleNum > 25:
    break

cap.release()
conn.commit()
conn.close()
cv2.destroyAllWindows()
