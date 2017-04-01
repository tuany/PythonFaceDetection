from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def detect(args):
	# load the image, convert it to grayscale, and blur it slightly
	print("Reference stripe detect module")
	image = cv2.imread(args["image"])
	print("--Converting to gray scale")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("--Applying Gaussian Blur")
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	print("--Detecting edges with Canny")
	edged = cv2.Canny(gray, 200, 400)
	print("----Performing dilation and erosion...")
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# find contours in the edge map
	print("--Finding countours")
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	print("----Sorting countours: top-to-bottom")
	(cnts, _) = contours.sort_contours(cnts, "top-to-bottom")
	pixelsPerMetric = None

	# I will only use the most top contourArea that is the reference stripe
	c = cnts[0]
	i = 1
	while cv2.contourArea(c) < 100 and i < len(cnts):
	# if the contour is not sufficiently large, ignore it
		c = cnts[i]
		i=i+1
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 2, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	print("----Calculating coordinates")
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 1)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 1)

	print("----Computing euclidean distance between points in the reference stripe...")
	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, centimeters)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	print("--Reference stripe width in pixels (%.2f) and centimeters (%.2f)" % (dB, dimB))

	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}cm".format(dimB),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.55, (0, 25, 255), 1)
	cv2.putText(orig, "{:.1f}cm".format(dimA),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.55, (0, 25, 255), 1)

	# show the output image
	cv2.imshow("Image", orig)
	cv2.waitKey(0)

	return { "w-pixels": dB, "w-centimeters": dimB, "h-pixels": dA, "h-centimeters": dimA, "pixelsPerMetric": pixelsPerMetric, "coordinates": [tl, tr, br, bl] }

if __name__ == '__main__':
	print("Detect a reference stripe in top of the image module")