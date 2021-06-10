import os
import cv2
import numpy as np
from PIL import Image

#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/LBPH.yml")

path = "yalefaces/teste"

detectorface = cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')

totalhits = 0.0
percentCorrect = 0.0
entireConfidence = 0.0

imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
for imagePath in imagePaths:
    faceImg = Image.open(imagePath).convert('L')
    faceNp = np.array(faceImg,'uint8')
    #equalized = cv2.equalizeHist(faceNp)
    detectFaces = detectorface.detectMultiScale(faceNp)

    for (x,y,w,h) in detectFaces:
        PrevID, Confidence = recognizer.predict(faceNp)
        AtID = int(os.path.split(imagePath)[1].split(".")[0].replace("subject",""))
        print(str(AtID) + " foi classificado como " + str(PrevID) + " - " + str(Confidence))

        if PrevID == AtID:
            totalhits += 1
            entireConfidence += Confidence    

print(totalhits)
percentCorrect = (totalhits / 30) * 100
totalConfidence = entireConfidence / totalhits

print("Total de acertos: " + str(percentCorrect))
print('Total de confiança: ' + str(totalConfidence))
