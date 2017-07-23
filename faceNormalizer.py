################################# imports
import math
import cv2
import numpy as np
from PIL import Image
import cmath
from decimal import *
import os
import config as cf
import logger

def eyeCoordinate(img, h, w1, w2):
    sumX = 0
    sumY = 0
    countX = 0
    countY = 0
    for i in range(w1, w2):
        for j in range(0, h):
            px = img[j, i]
            ent = px.astype(np.int)
            if((ent <= 275) & (ent >= 250)):
                sumX = sumX + i
                sumY = sumY + j
                countX = countX + 1
                countY = countY + 1
    x = sumX / countX
    y = sumY / countY    
    return y, x

def normalize(fn):
    log = logger.getLogger(__file__)
    ################################# load image
    img_folder = fn+"_output"
    if not os.path.exists(img_folder):
        log.info("Creating output images folder {}".format(img_folder))
        os.makedirs(img_folder)
    img_path = fn
    if os.path.isfile(img_folder + "/undistorted.jpg"):
        img_path = img_folder + "/undistorted.jpg"
    rotated_img_path = img_folder + "/rotated.jpg" 
    img = cv2.imread(img_path)
    log.info("Starting pre-processing of image {}".format(img_path))

    colorful = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
    log.debug("Loading cascade classifier: %s" % "haarcascade_frontalface_alt2.xml")
    ################################# detect face
    log.debug("Trying to find a face:")
    faces = face_cascade.detectMultiScale(gray, cf.SCALE_FACTOR, cf.MIN_NEIGHBORS, cv2.CASCADE_SCALE_IMAGE, cf.MIN_SIZE)
    log.info("Found {0} faces!".format(len(faces)))
    for (x,y,w,h) in faces:
        rectangle = cv2.rectangle(colorful,(x,y),(x+w,y+h),(255,0,0),4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = colorful[y:y+h, x:x+w]
    ################################# cut image
    log.info("Cropping detected face:")
    image = Image.open(img_path) 
    box = (x, y, x + w, y + h)
    face_box = image.crop(box)
    face = np.asarray(face_box)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_folder+"/cropped.jpg", face_gray)
    log.info("" + img_folder+"/cropped.jpg"+ " created!")

    ################################# detect eyes pair
    eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_mcs_eyepair_big.xml')
    log.debug("Loading cascade classifier: %s" % "haarcascade_mcs_eyepair_big.xml")
    eyes = eye_cascade.detectMultiScale(face_gray)
    log.debug("Trying to find eyes pair:")
    log.info("Found {0} eyes pair!".format(len(eyes)))

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(face_gray, (ex,ey),(ex+ew,ey+eh),(0,255,0), 2)
        eyes_pair = face_gray[ey:ey+eh,ex:ex+ew]
        eyes_box = (ex,ey,ex+ew,ey+eh)
        
    cv2.imwrite(img_folder+"/eyes.jpg", eyes_pair)
    log.info("" + img_folder+"/eyes.jpg"+ " created!")

    ################################# detect eyes borders
    log.debug("Detecting eyes borders")
    canny = cv2.Canny(eyes_pair, 120, 245)
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)
    gradient_path = img_folder + "/eyes_gradient.jpg"
    cv2.imwrite( gradient_path , gradient )
    ################################# find coordinates of the pupils
    h = gradient.shape[0]
    w = gradient.shape[1]
    ################################# normalize image geometrically
    y1, x1 = eyeCoordinate(gradient, h, 0, w/2) # left eye
    log.debug("Normalizing geometrically:")
    y1 = abs(y1)
    x1 = abs(x1)
    y2, x2 = eyeCoordinate(gradient, h, w/2, w) # right eye
    y2 = abs(y2)
    x2 = abs(x2)
    dy = y2 - y1
    dx = x2 - x1
    dy = abs(dy)
    dx = abs(dx)
    dy = dy*(-1)
    z = Decimal(dy) / Decimal(dx)
    alpha_complex = cmath.atan(z)
    alpha = cmath.phase(alpha_complex)
    alpha = alpha / 2
    log.debug("Alpha angle to rotate image: %f" % alpha)
    log.info("Rotating and saving output image:")
    img = Image.open(img_path)
    rotated_image = img.rotate(-alpha)
    rotated_image.save(rotated_img_path)

    log.debug("Output image: %s saved!" % rotated_img_path)

if __name__ == '__main__':
    print("Image pre-processor module")