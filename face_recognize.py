# ---------------------- Recogniser for LBPH Recogniser ------------------------------------------ #
# ------------------------------ By Mr-maike ----------------------------------------------------- #
import cv2
import numpy as np
import RPi.GPIO as GPIO
import sqlite3
import os

#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)
#GPIO.setup(18, GPIO.OUT)

conn = sqlite3.connect('usersdatabase.db')
c = conn.cursor()
fname = "model/trainingData.yml"
 
if not os.path.isfile(fname):
  print("Por favor, treine o reconhecedor primeiro!")
  exit(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)

while True:

  _,img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
    c.execute("select name from users where id = (?);", (ids,))
    result = c.fetchall()
    name = result[0][0]

    if conf < 50:
      cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
      print('[SISTEMA] Seja bem-vindo ' + str(name))
      #GPIO.output(18, GPIO.HIGH)
      
    else:
      cv2.putText(img, 'Desconhecido', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
      #GPIO.output(18, GPIO.LOW)

  cv2.imshow('Reconhecimento Facial',img)
   
  k = cv2.waitKey(30) & 0xff 
  if k == 27:
    #GPIO.output(18, GPIO.LOW)  
    break

cap.release()
cv2.destroyAllWindows()