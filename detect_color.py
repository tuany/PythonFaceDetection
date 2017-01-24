# based on tutorial available at: pyimagesearch.com/2014/08/04/opencv-python-color-detection/

# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# define the list of boundaries
boundaries = [
	([40, 50, 140], [100, 100, 250])
]

print("Detectando faixa no limite: ")
print(boundaries)

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	print(output.shape)
	# show the images
cv2.imshow("Mascara", np.hstack([image, output]))
output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# constant cv2.CV_CHAIN_APPROX_NONE = 1
# constant cv2.RETR_LIST            = 1
contours,_ = cv2.findContours(output, 0, 2)

lst_intensities = []

for i in range(len(contours)):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(output)
    cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    lst_intensities.append(output[pts[1], pts[0]])

cv2.waitKey(0)