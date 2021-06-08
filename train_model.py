import os
import cv2
import numpy as np 
from PIL import Image

recognizerEigenfaces = cv2.face.EigenFaceRecognizer_create()
recognizerFisherFaces = cv2.face.FisherFaceRecognizer_create(2)
recognizerLBPH = cv2.face.LBPHFaceRecognizer_create(1, 8,8,8)

path = 'dataset'

if not os.path.exists('./model'):
    os.makedirs('./model')
 
def getImagesWithID(path):
  imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
  faces = []
  IDs = []
  for imagePath in imagePaths:
    faceImg = Image.open(imagePath).convert('L')
    faceNp = np.array(faceImg,'uint8')
    ID = int(os.path.split(imagePath)[-1].split('.')[1])
    faces.append(faceNp)

    IDs.append(ID)
    cv2.imshow("training",faceNp)
    cv2.waitKey(10)
  return np.array(IDs), faces
Ids, faces = getImagesWithID(path)

print('[SISTEMA] Treinando...')
recognizerEigenfaces.train(faces, Ids)
print('[SISTEMA] Arquivo Eigenface treinado com sucesso! ') 
recognizerEigenfaces.save('model/trainingDataEigenface.xml')

recognizerLBPH.train(faces, Ids)
print('[SISTEMA] Arquivo  LBPH treinado com sucesso! ')
recognizerLBPH.save('model/trainingDataLBPH.xml')


print('[SISTEMA] Todos os algoritmos foram criados com sucesso!')
cv2.destroyAllWindows()
