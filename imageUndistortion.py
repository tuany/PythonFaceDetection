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

def undistort(img_folder):
	image_path=img_folder
	img = cv2.imread(image_path)
	outfile=img_folder+"_output/undistorted.jpg"
	if not os.path.exists(img_folder):
		print("Creating output images folder {}".format(img_folder))
		os.makedirs(img_folder)

	print('Undistorting image: %s' % image_path)

	if not path.isfile(cf.DISTORTION_MATRIX):
		cameraCalibration02.calculateDistortionMatrix()

	dist_matrix = pkl.load(open(cf.DISTORTION_MATRIX, 'rb'))
	# try:
	h,  w = img.shape[:2]
	camera_matrix = dist_matrix[cf.CAMERA_MATRIX]
	print('Camera matrix')
	print('')
	print(camera_matrix)
	print('')
	print('')
	# dist_coefs = dist_matrix[cf.DISTANCE_COEFFICIENTS]
	dist_coefs = np.array([[-0.13755888, -1.07424959, -0.01249778, -0.00225594,  2.00720682]])
	print('Coefficients')
	print('')
	print(dist_coefs)
	print('')
	print('')
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0.75, (w, h))
	print('New camera matrix')
	print('')
	print(newcameramtx)
	print('')
	print('')
	print('ROI')
	print(roi)
	print('')
	# mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, newcameramtx, (w,h), 5)
	# new_image = cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)

	new_image = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
	# crop and save the image
	# x, y, w, h = roi
	# new_image = new_image[y:y+h, x:x+w]
	print('Undistorted image written to: %s' % outfile)
	cv2.imwrite(outfile, new_image)
	# except KeyError as e:
	# 	print("unable to find some key in %s" % cf.DISTORTION_MATRIX)
	# except IOError as ioe:
	# 	if ioe.errno == errno.EACCES:
	# 		print("--No write permittion")
	# 		os.chdir("../")
	# 	# Not a permission error.
	# 	raise

if __name__ == '__main__':
	print("Image undistortion module")