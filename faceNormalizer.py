################################# imports
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cmath
from decimal import *

################################# Carregando funcoes
print("Construindo funcoes...")
def leftEyeCoordinate(img, h, w):
    print("Coordenadas do olho esquerdo.")
    sumX = 0
    sumY = 0
    countX = 0
    countY = 0
    for i in range(0, w/2):
        for j in range(0, h):
            px = img[j, i]
            ent = px.astype(np.int)
            if((ent <= 275) & (ent >= 250)):
                sumX = sumX + i
                sumY = sumY + j
                countX = countX + 1
                countY = countY + 1
    print("Coordenadas do olho esquerdo..")
    x = sumX / countX
    y = sumY / countY    
    print(x, y)
    print("Coordenadas do olho esquerdo... OK")
    return y, x

def rightEyeCoordinate(img, h, w):
    print("Coordenadas do olho direito.")
    sumX = 0
    sumY = 0
    countX = 0
    countY = 0
    for i in range(w/2, w):
        for j in range(0, h):
            px = img[j, i]
            ent = px.astype(np.int)
            if((ent <= 275) & (ent >= 250)):
                sumX = sumX + i
                sumY = sumY + j
                countX = countX + 1
                countY = countY + 1
    print("Coordenadas do olho direito..")
    x = sumX / countX
    y = sumY / countY    
    print(x, y)
    print("Coordenadas do olho direito... OK")
    return y, x

print("Construindo funcoes... OK")
################################# load image
print("Carregando imagem...")
img_path = './img/MIT-9.jpg'
rotated_img_path = './img/MIT-9-rotated.jpg'
img = cv2.imread(img_path)

colorful = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Carregando imagem... OK")

face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
################################# detect face
print("Detectando face...")
faces = face_cascade.detectMultiScale(gray, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20,20))
for (x,y,w,h) in faces:
    print((x,y,w,h))
    rectangle = cv2.rectangle(colorful,(x,y),(x+w,y+h),(255,0,0),4)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = colorful[y:y+h, x:x+w]
    
    print(roi_gray.shape)

print("Detectando face... OK")
################################# cut image
print("Recortando...")
image = Image.open(img_path) 
box = (x,y,x+w,y+h)
face_box = image.crop(box)
face = np.asarray(face_box)
face_gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
print("Recortando... OK")
################################# detect eyes pair
eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_mcs_eyepair_big.xml')
eyes = eye_cascade.detectMultiScale(face_gray)
print("Encontrando os olhos...")

for (ex,ey,ew,eh) in eyes:
    print((ex,ey,ew,eh))
    cv2.rectangle(face, (ex,ey),(ex+ew,ey+eh),(0,255,0), 2)
    eyes_pair = face_gray[ey:ey+eh,ex:ex+ew]
    
cv2.imwrite('./img/MIT-9/eyes.jpg', eyes_pair)
global local_olho_x 
local_olho_x = ex
global local_olho_y
local_olho_y = ey
print("Encontrando os olhos... OK")
################################# detect eyes borders
print("Detectando a borda dos olhos...")
canny = cv2.Canny(eyes_pair, 120, 245)
kernel = np.ones((3, 3), np.uint8)
gradiente = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite('./img/MIT-9/eyes_gradiente.jpg', gradiente)
print("Detectando a borda dos olhos... OK")
################################# find coordinates of the pupils
h = gradiente.shape[0]
w = gradiente.shape[1]
################################# normalize image geometrically
y1, x1 = leftEyeCoordinate(gradiente, h, w)
print("Iniciando normalizacao geometrica...")
y1 = abs(y1)
x1 = abs(x1)
y2, x2 = rightEyeCoordinate(gradiente, h, w)
y2 = abs(y2)
x2 = abs(x2)
print(x1, y1, x2, y2)
dy = y2 - y1
dx = x2 - x1
dy = abs(dy)
dx = abs(dx)
dy = dy*(-1)
z = Decimal(dy) / Decimal(dx)
alfa_complexo = cmath.atan(z)
alfa = cmath.phase(alfa_complexo)
alfa = alfa / 2
print(alfa)
print("Normalizando imagem geometricamente... OK")
################################# rotate image
img = Image.open(img_path)
rotated_image = img.rotate(-alfa)
rotated_image.save(rotated_img_path)
print("Salvando imagem rotacionada... OK")
rotated_image.show()
c = cv2.waitKey()
cv2.destroyAllWindows()
