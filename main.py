
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
import calculateDistances
import config as cf
import logger


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

def aggregateDistances(distances, all_img_distances, output, idt, fn):
	distances["id"] = idt
	distances["img_name"] = fn
	all_img_distances[idt] = distances
	file_exists = os.path.isfile(output)
	with open (output+".csv", 'wb') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=distances.keys())
		if not file_exists:
			writer.writeheader()
		writer.writerow(distances)
	with open (output+".pkl", 'wb') as pklfile:
		pickle.dump(distances, pklfile)
	return all_img_distances

if __name__ == '__main__':
	# initialization and logging configuration
	start_time = time.time()
	log = logger.getLogger(__file__)
	log.info("Image processing started")
	log.info("Switching to dir %s " % cf.ROOT_DIR)
	os.chdir(cf.ROOT_DIR)

	# get all images in img/ directory 
	img_names = glob(cf.INPUT_IMG_MASK)
	img_names_processed = []
	few_distances_dict = {}
	all_distances_dict = {}
	farkas_distances_dict = {}
	log.info("%4d images found!" % len(img_names))
	count = 1
	for fn in img_names:
		log.info('processing %s... ' % fn)
		output_folder = fn+"_output"
		img = cv2.imread(fn, 0)

		if img is None:
			log.warning("Failed to load", fn)
			continue

		final_image_path = os.path.abspath(output_folder+"/rotated.jpg")

		file_exists = os.path.isfile(final_image_path)
		reference_file_exists = os.path.isfile(output_folder+"/"+"reference_stripe.pkl")
		if not file_exists:
			iu.undistort(fn)
		try:
			reference_info = {}
			if not file_exists:
				faceNormalizer.normalize(fn)
			if not reference_file_exists:
				reference_info = detectReferenceStripe.detect(final_image_path)	
				try:
				  reference_stripe_file = output_folder+"/"+"reference_stripe"
				  csvfile = open(reference_stripe_file+".csv", 'wb')
				  fieldnames = ['w-pixels', 'w-centimeters', 'h-pixels', 'h-centimeters', 'pixelsPerMetric', 'coordinates']
				  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
				  writer.writeheader()
				  writer.writerow(reference_info)
				  with open(reference_stripe_file+".pkl", "wb") as f:
				  	pickle.dump(reference_info, f)
				except IOError as e:
				    if e.errno == errno.EACCES:
				        log.error("--No write permittion")
				        os.chdir(cf.ROOT_DIR)
				    else:
					    # Not a permission error.
					    raise
				finally:
					csvfile.close()
			else:
				with open(output_folder+"/"+"reference_stripe.pkl", "rb") as f:
					reference_info = pickle.load(f)
		except UnboundLocalError as ule:
			log.exception("Failed to process %s", fn)
			continue
		except cv2.error as cverr:
			log.exception("Exception processing %s", fn)
			continue
		except ValueError as ve:
			log.exception("Exception processing %s", fn)
			continue

		# faceNormalizer.normalize( fn )
		log.info("Calling external programming for feature extraction")
		file_exists = os.path.isfile(cf.ROOT_DIR+"/"+output_folder+"/points.csv")
		if not file_exists:
			process_out = subprocess.check_call([cf.EXTERNAL_EXEC_DIR+"/FeatureExtraction", "-f", final_image_path, "-of", output_folder+"/points.csv", "-oi", output_folder+"/marked.jpg", "-no3Dfp", "-noMparams", "-noPose", "-noAUs", "-noGaze"], shell=True)
		else:
			process_out = 0
		if process_out == 0:
			log.info("Feature extraction done. Writing results to file.")
			points_dict = {}
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

			distances_few = calculateDistances.few(final_image_path, "img_"+str(count), points_dict, reference_info)
			few_distances_dict = aggregateDistances(distances_few, few_distances_dict, cf.ROOT_DIR+"/"+output_folder+"/"+"distances_few", count, fn)
			
			distances_all = calculateDistances.all(points_dict, reference_info)
			all_distances_dict = aggregateDistances(distances_all, all_distances_dict, cf.ROOT_DIR+"/"+output_folder+"/"+"distances_all", count, fn)

			distances_farkas = calculateDistances.farkas(points_dict, reference_info)
			farkas_distances_dict = aggregateDistances(distances_farkas, farkas_distances_dict, cf.ROOT_DIR+"/"+output_folder+"/"+"distances_farkas", count, fn)

			count = count + 1 
			img_names_processed.append(fn)
		else:
			log.critical("Landmarking extraction did not went well! Finishing image processing...")
			continue
		os.chdir(cf.ROOT_DIR)
	with open(cf.OUTPUT_DIR+"/"+"distances_few.pkl", "wb") as f:
	    pickle.dump(len(few_distances_dict), f)
	    for value in few_distances_dict:
	        pickle.dump(value, f)
	with open(cf.OUTPUT_DIR+"/"+"distances_farkas.pkl", "wb") as f:
	    pickle.dump(len(farkas_distances_dict), f)
	    for value in farkas_distances_dict:
	        pickle.dump(value, f)
	with open(cf.OUTPUT_DIR+"/"+"distances_all.pkl", "wb") as f:
	    pickle.dump(len(all_distances_dict), f)
	    for value in all_distances_dict:
	        pickle.dump(value, f)
	log.info("Total: %d " % len(img_names_processed))	
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))
