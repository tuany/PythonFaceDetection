''' 
	undistort input image with parameters previously recorded or
	using the script cameraCalibration if parameters not available
'''

import cv2
import os
import os.path as path
import config as cf
# import csv
import pickle as pkl
import numpy as np
import cameraCalibration02
import logger

def undistort(img_folder):
	log = logger.getLogger(__file__)
	image_path=img_folder
	img = cv2.imread(image_path)
	outfile=img_folder+"_output/undistorted.jpg"
	if not os.path.exists(img_folder):
		log.info("Creating output images folder {}".format(img_folder))
		os.makedirs(img_folder)

	log.info("Undistorting image: %s" % image_path)

	if not path.isfile(cf.DISTORTION_MATRIX):
		log.info('Distortion matrix file not found. Calculating new matrix.')
		cameraCalibration02.calculateDistortionMatrix()

	dist_matrix = pkl.load(open(cf.DISTORTION_MATRIX, 'rb'))
	h,  w = img.shape[:2]
	camera_matrix = dist_matrix[cf.CAMERA_MATRIX]
	dist_coefs = np.array([[-0.13755888, -1.07424959, -0.01249778, -0.00225594,  2.00720682]])
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0.75, (w, h))
	new_image = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
	log.info("Undistorted image written to: %s" % outfile)
	cv2.imwrite(outfile, new_image)
	
if __name__ == '__main__':
	print("Image undistortion module")