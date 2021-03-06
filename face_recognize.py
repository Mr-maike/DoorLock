# ---------------------- Recogniser for LBPH Recogniser ------------------------------------------ #
# ------------------------------ By Mr-maike ----------------------------------------------------- #
import cv2
import numpy as np
import sqlite3
import os

conn = sqlite3.connect('usersdatabase.db')
c = conn.cursor()
fname = "model/trainingDataLBPH.yml"
 
if not os.path.isfile(fname):
  print("Por favor, treine o reconhecedor primeiro!")
  exit(0)

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)

while True:

  _,img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.1, 5)

  for (x,y,w,h) in faces:

    gray_face = gray[y: y+h, x: x+w]
    eyes = eye_cascade.detectMultiScale(gray_face)

    for(ex, ey, ew, eh) in eyes:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

      ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
      c.execute("select name from users where id = (?);", (ids,))
      result = c.fetchall()
      name = result[0][0]

      if conf < 12:
        cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
        cv2.putText(img, str(conf), (x,y + (h+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0), 2)
        print('[SISTEMA] Seja bem-vindo ' + str(name))
      
      else:
        cv2.putText(img, 'Desconhecido', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

  cv2.imshow('Reconhecimento Facial',img)
   
  k = cv2.waitKey(30) & 0xff 
  if k == 27:
    break

cap.release()
cv2.destroyAllWindows()