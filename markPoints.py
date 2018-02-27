# Python 2/3 compatibility
from __future__ import print_function
from glob import glob

# python modules
import argparse
import csv
import os
import subprocess
import time
import pickle

# external libraries
import cv2

# custom scripts
import imageUndistortion as iu
import detectReferenceStripe
import faceNormalizer
import calculateDistancesCm
import calculateDistancesPx
import config as cf
import logger
import copy


'''
	print points
'''

if __name__ == '__main__':
	# initialization and logging configuration
	start_time = time.time()
	log = logger.getLogger(__file__)
	log.info("Image processing started")
	log.info("Switching to dir %s " % cf.ROOT_DIR)
	os.chdir(cf.ROOT_DIR)

	# get all images in img/ directory 
	img_names = glob('img/MIT*.jpg')
	img_names_processed = []
	
	log.info("%4d images found!" % len(img_names))
	count = 1
	for fn in img_names:
		log.info('processing %s... ' % fn)
		output_folder = fn+"_output"
		img = cv2.imread(fn, 0)

		if img is None:
			log.warning("Failed to load", fn)
			continue

		final_image_path = os.path.abspath(output_folder+img)

		log.info("Calling external programming for feature extraction")
		file_exists = os.path.isfile(cf.ROOT_DIR+"/"+output_folder+"/points.csv")
		if not file_exists:
			process_out = subprocess.check_call([cf.EXTERNAL_EXEC_DIR+"/FeatureExtraction", "-f", final_image_path, "-of", output_folder+"/points.csv", "-oi", output_folder+"/marked.jpg", "-no3Dfp", "-noMparams", "-noPose", "-noAUs", "-noGaze"], shell=True)
		else:
			process_out = 0
		if process_out == 0:
			log.info("Feature extraction done. Writing results to file.")
			points_dict = {}
			turnedoff = False
			if not turnedoff:
				try:
					os.chdir(cf.ROOT_DIR+"/"+output_folder)
					points_file = open("points.csv", 'rb')
					reader = csv.reader(points_file)
					headers = reader.next()
					for row in reader:
						for h, v in zip(headers, row):
							points_dict[h.strip()] = float(v)
					log.info("Points file saved!")
					points_file.close()
					with open("points.pkl", "wb") as f:
				  		pickle.dump(points_dict, f)

				except NameError as nme:
					log.exception("Points file not found!")

			    refCoords = []
				for (i, j) in izip(range(0, 68), range(0, 68)):
					x1 = "x_" + str(i)
					y1 = "y_" + str(j)
					refCoords.append((points_dict[x1], points_dict[y1]))

				refCoords = np.vstack(refCoords)
				names = []
				# refCoords = np.vstack([])

				for key, value in points_dict.iteritems():
					names.append(key)
				# plot
				log.info("Image path {}".format(final_image_path))
				image = cv2.imread(final_image_path)
				color = (0, 0, 255) # vermelho
				if image is not None :
					orig = image.copy()
					# loop over the original points
					for ((u, v), (w, z), nam) in zip(refCoords, objCoords, names):
						# draw circles corresponding to the current points and
						# connect them with a line
						color = randomColor()
						cv2.circle(orig, (int(u), int(v)), 4, color, -1)
						cv2.circle(orig, (int(w), int(z)), 4, color, -1)
						cv2.line(orig, (int(u), int(v)), (int(w), int(z)),
						color, 1)
						D = dist.euclidean((u, v), (w, z)) / pixelsPerMetric
						(mX, mY) = midpoint((u, v), (w, z))
						cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

						# show the output image
					cv2.imwrite(cf.OUTPUT_DIR + "/" + output_name + ".jpg", orig)
			count = count + 1 
			img_names_processed.append(fn)
		else:
			log.critical("Landmarking extraction did not went well! Finishing image processing...")
			continue
		os.chdir(cf.ROOT_DIR)
	log.info("Total: %d " % len(img_names_processed))	
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))
