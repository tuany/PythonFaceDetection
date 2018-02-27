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

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def eye_coordinate(img, h, w1, w2):
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
    orig = cv2.imread(img_path)
    log.info("Starting pre-processing of image {}".format(img_path))

    colorful = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

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
    canny = cv2.Canny(eyes_pair, 50, 245)
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)
    gradient_path = img_folder + "/eyes_gradient.jpg"
    cv2.imwrite( gradient_path , gradient )
    ################################# find coordinates of the pupils
    h = gradient.shape[0]
    w = gradient.shape[1]
    ################################# normalize image geometrically
    y1, x1 = eye_coordinate(gradient, h, 0, w/2) # left eye
    log.debug("Normalizing geometrically:")
    y1 = abs(y1)
    x1 = abs(x1)
    y2, x2 = eye_coordinate(gradient, h, w/2, w) # right eye
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

    log.info("Cropping and saving final normalized image")
    image_height, image_width = orig.shape[0:2]

    image_rotated = cv2.imread(rotated_img_path)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(-alpha)
        )
    )

    cv2.imwrite(img_folder+"/cropped.jpg", image_rotated_cropped)
    log.debug("Output image: %s saved!" % image_rotated_cropped)

if __name__ == '__main__':
    print("Image pre-processor module")