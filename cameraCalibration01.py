#!/usr/bin/python
import cv2
import sys
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
DIM_X = 9
DIM_Y = 6
chessboard_dim = (DIM_X, DIM_Y)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((DIM_Y*DIM_X,3), np.float32)
objp[:,:2] = np.mgrid[0:DIM_X,0:DIM_Y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('data/chessboard0*.jpg')
print(images)
count = 1
for fname in images:
	print("Loading first image: ", fname)
	gray = cv2.imread(fname, 0)
	img_color = cv2.imread(fname)
	print("Trying to find corners: ")
	found_all, corners = cv2.findChessboardCorners( gray, chessboard_dim )
	if found_all == True:
		print("Found: ", found_all)
		objpoints.append(objp)
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)
		print("Drawint points in the chessboard...")
		cv2.drawChessboardCorners( img_color, chessboard_dim, corners, found_all )
		# cv2.imshow("Chessboard with corners: "+ fname, img_color)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	img = cv2.imread(fname)
	h,  w = img.shape[:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

	# undistort
	mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
	dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

	# crop
	x,y,w,h = roi
	dst = dst[y:y+h, x:x+w]

	output = "./output/"+str(count)+".png"
	cv2.imwrite(output,dst)

	tot_error = 0
	for i in xrange(len(objpoints)):
		imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		tot_error += error

	print "total error: ", tot_error/len(objpoints)
	count=count+1

cv2.waitKey()