
# Python 2/3 compatibility
from __future__ import print_function
from glob import glob

import argparse
import imageUndistortion as iu
import detectReferenceStripe
import faceNormalizer
import calculateDistances
import csv
import os
import subprocess
import config as cf
import time
import cv2


'''
	This is the main script that try to obtain facial distances of a 
	photo taken using the protocol defined by Tuany Dias Pinheiro.

	This is the pipeline:

	1 - Camera calibration 
	2 - Photo normalizer (rotation)
	3 - Reference stripe detection (obtain reference coordinates and size)
	4 - Points extraction
	5 - Distances calculus
'''

if __name__ == '__main__':
	start_time = time.time()
	# initialization
	root_path = os.path.abspath(cf.ROOT_DIR)
	# get all images in img/ directory 
	img_names = glob(cf.INPUT_IMG_MASK)
	img_names_processed = []
	print("%4d images found!" % len(img_names), end='')
	print("ok")
	for fn in img_names:
		print('processing %s... ' % fn, end='')
		output_folder = fn+"_output"
		img = cv2.imread(fn, 0)

		if img is None:
			print("Failed to load", fn)
			continue

		final_image_path = output_folder+"/rotated.jpg"

		# first detect the stripe
		faceNormalizer.normalize(fn)
		reference_info = detectReferenceStripe.detect(final_image_path)
		# print("Saving reference_stripe results")
		# try:
		#   csvfile = open(output_folder+"/"+"reference_stripe.csv", 'wb')
		#   fieldnames = ['w-pixels', 'w-centimeters', 'h-pixels', 'h-centimeters', 'pixelsPerMetric', 'coordinates']
		#   writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		#   writer.writeheader()
		#   writer.writerow(reference_info)
		# except IOError as e:
		#     if e.errno == errno.EACCES:
		#         print("--No write permittion")
		#         os.chdir("../")
		#     # Not a permission error.
		#     raise

		iu.undistort(fn)
		faceNormalizer.normalize( fn )

		process_out = subprocess.check_call([cf.EXTERNAL_EXEC_DIR+"/FeatureExtraction", "-f", final_image_path, "-of", output_folder+"/points.csv", "-oi", output_folder+"/marked.jpg", "-no3Dfp", "-noMparams", "-noPose", "-noAUs", "-noGaze"], shell=True)
		if process_out == 0:
			points_dict = {}
			try:
				os.chdir(root_path+"/"+output_folder)
				points_file = open("points.csv", 'rb')
				reader = csv.reader(points_file)
				headers = reader.next()
				for row in reader:
					for h, v in zip(headers, row):
						points_dict[h.strip()] = float(v)   
			finally:
				points_file.close()
			distances = calculateDistances.farkas(points_dict, reference_info)
			csvfile = open("distances.csv", 'w')
			writer = csv.DictWriter(csvfile, fieldnames=distances.keys())
			writer.writeheader()
			writer.writerow(distances)
		else:
			raise Exception('Landmarking extraction did not went well! Finishing image processing...')
		img_names_processed.append(fn)
		print('ok')
		os.chdir(root_path)
	print("Images processed: ")	
	print(img_names_processed)
	print("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))