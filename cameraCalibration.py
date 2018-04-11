#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images
usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]
default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
from common import splitfn

# built-in modules
import os
# import csv
import pickle

import sys
import getopt
from glob import glob
import config as cf
import logger

def calculateDistortionMatrix():
    log = logger.getLogger(__file__)
    img_mask = 'data/chessboard*.jpg'  # default
    log.info("Using image mask {} in dir".format(img_mask))
    img_names = glob(img_mask)
    debug_dir = 'output/'
    if not os.path.isdir(debug_dir):
        log.info("Creating dir {}".format(debug_dir))
        os.mkdir(debug_dir)
    
    square_size = cf.CHESSBOARD_SQUARE_SIZE # cm
    search_window = cf.CHESSBOARD_SEARCH_WINDOW
    pattern_size = cf.CHESSBOARD_PATTERN_SIZE

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    for fn in img_names:
        log.info('processing %s... ' % fn)
        img = cv2.imread(fn, 0)
        if img is None:
            log.error("Failed to load {}. Going to the next image.".format(fn))
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, search_window, (-1, -1), term)

        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            outfile = debug_dir + name + '_chess.png'
            cv2.imwrite(outfile, vis)
            if found:
                img_names_undistort.append(outfile)

        if not found:
            log.warning('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)
        log.info("Chessboard found in image {}. Processed!".format(fn))

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    log.debug("\nRMS:", rms)
    log.debug("camera matrix:\n", camera_matrix)
    log.debug("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    log.info("Testing the distortion matrix")
    for img_found in img_names_undistort:
        img = cv2.imread(img_found)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        outfile = img_found + '_undistorted.png'
        log.info('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

    log.info("Saving distortion matrix")
    reference_info = { "rms": rms, "camera_matrix": camera_matrix, "dist_coefs": dist_coefs }
    try:
        root_path = os.getcwd();
        output_folder = root_path+"/"+"output"
        if not os.path.exists(output_folder):
            log.info("Creating output folder {}".format(output_folder))
            os.makedirs(output_folder)

        pickle.dump(reference_info, open(cf.DISTORTION_MATRIX, 'wb'))
        log.info("Saved distortion matrix {}".format(cf.DISTORTION_MATRIX))
    except IOError as e:
        if e.errno == errno.EACCES:
            log.exception("--No write permittion")
            os.chdir("../")
        # Not a permission error.
        raise

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Calculating distortion matrix")
    calculateDistortionMatrix()