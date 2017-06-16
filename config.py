'''
	Program configs and constants
'''

import os

# common directories
ROOT_DIR=os.getcwd()
OUTPUT_DIR=ROOT_DIR+"/output"
IMG_DIR=ROOT_DIR+"/img"
TRAINING_DATA_DIR=ROOT_DIR+"/data"
EXTERNAL_EXEC_DIR=ROOT_DIR+"/exec"
CASCADE_CLASSIFIERS_DIR=ROOT_DIR+"/classifiers"
DISTORTION_MATRIX=OUTPUT_DIR+"/distortion_matrix.pkl"

# measure units and vals
REFERENCE_STRIPE_WIDTH=50 #cm
REFERENCE_STRIPE_HEIGTH=5 #cm
CHESSBOARD_SQUARE_SIZE=3 #cm
CHESSBOARD_SEARCH_WINDOW=(6, 6)
CHESSBOARD_PATTERN_SIZE=(9, 6)
SCALE_FACTOR=1.3 # face cascade detect mustiscale scale factor
MIN_NEIGHBORS=5 # face cascade min neighbors
MIN_SIZE=(20,20) # face cascade window size
NORM_FACTOR=1.6934931395 # normalization factor. The face is not in the same plan as the stripe so this factor was calculated

# script for camera calibration
CAMERA_CALIBRATION_SCRIPT="cameraCalibration02"

# keys
PIXELS_PER_METRIC="pixels_per_metric"
RMS="rms"
CAMERA_MATRIX="camera_matrix"
DISTANCE_COEFFICIENTS="dist_coefs"
FACE_CASCADE_CLASSIFIER=CASCADE_CLASSIFIERS_DIR+"/haarcascade_frontalface_alt2.xml"

if __name__ == '__main__':
    print("Configurations and constants")