import os
import cv2
import numpy as np 
from PIL import Image

Eigenfaces = cv2.face.EigenFaceRecognizer_create(15)
FisherFaces = cv2.face.FisherFaceRecognizer_create(2)
lbph = cv2.face.LBPHFaceRecognizer_create(1, 1,7,7)


path = 'yalefaces/treinamento'

#if not os.path.exists('./model'):
#    os.makedirs('./model')
 
def getImagesWithID(path):
  imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
  faces = []
  IDs = []
  for imagePath in imagePaths:
    faceImg = Image.open(imagePath).convert('L')
    faceNp = np.array(faceImg,'uint8')
    ID = int(os.path.split(imagePath)[1].split('.')[0].replace("subject",""))
    faces.append(faceNp)

    IDs.append(ID)
    cv2.imshow("training",faceNp)
    cv2.waitKey(10)
  return np.array(IDs), faces


Ids, faces = getImagesWithID(path)

print('[SISTEMA] Treinando...')
Eigenfaces.train(faces, Ids)
print('[SISTEMA] Arquivo Eigenface treinado com sucesso! ') 
Eigenfaces.save('model/Eigenface.yml')

FisherFaces.train(faces, Ids)
print('[SISTEMA] Arquivo Fisherface treinado com sucesso! ')
FisherFaces.write('model/FisherFace.yml')

lbph.train(faces, Ids)
print('[SISTEMA] Arquivo  LBPH treinado com sucesso! ')
lbph.save('model/LBPH.yml')


print('[SISTEMA] Todos os algoritmos foram criados com sucesso!')
cv2.destroyAllWindows()
