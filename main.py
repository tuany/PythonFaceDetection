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
	This is the main script that try to obtain facial distances of a 
	photo taken using the protocol defined by Tuany Dias Pinheiro.

	This is the pipeline:

	1 - Camera calibration 
	2 - Photo normalizer (rotation)
	3 - Reference stripe detection (obtain reference coordinates and size)
	4 - Points extraction
	5 - Distances calculus
'''

def aggregateDistances(distances_eu, all_img_distances_eu, distances_mh, all_img_distances_mh, output, idt, fn):
	distances_eu["id"] = idt
	distances_eu["img_name"] = fn
	all_img_distances_eu[idt] = distances_eu

	distances_mh["id"] = idt
	distances_mh["img_name"] = fn
	all_img_distances_mh[idt] = distances_mh

	output_eu = output + '_eu'
	output_mh = output + '_mh'

	file_exists = os.path.isfile(output_eu+'.csv')
	if file_exists:
		mode = 'ab'
	else:
		mode = 'wb'
	with open (output_eu+".csv", mode) as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=distances_eu.keys())
		if not file_exists:
			writer.writeheader()
		writer.writerow(distances_eu)
	log.info("Euclidian distances file saved in output dir!")
	with open (output_eu+".pkl", mode) as pklfile:
		pickle.dump(distances_eu, pklfile)

	with open (output_mh+".csv", mode) as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=distances_mh.keys())
		if not file_exists:
			writer.writeheader()
		writer.writerow(distances_mh)
	log.info("Manhattan distances file saved in output dir!")
	with open (output_mh+".pkl", mode) as pklfile:
		pickle.dump(distances_mh, pklfile)

	return all_img_distances_eu, all_distances_dict_mh

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
	few_distances_dict_eu = {}
	all_distances_dict_eu = {}
	farkas_distances_dict_eu = {}
	few_distances_dict_mh = {}
	all_distances_dict_mh = {}
	farkas_distances_dict_mh = {}

	few_distances_dict_eu_px = {}
	all_distances_dict_eu_px = {}
	farkas_distances_dict_eu_px = {}
	few_distances_dict_mh_px = {}
	all_distances_dict_mh_px = {}
	farkas_distances_dict_mh_px = {}

	few_distances_dict_eu_px_1000 = {}
	all_distances_dict_eu_px_1000 = {}
	farkas_distances_dict_eu_px_1000 = {}
	few_distances_dict_mh_px_1000 = {}
	all_distances_dict_mh_px_1000 = {}
	farkas_distances_dict_mh_px_1000 = {}

	log.info("%4d images found!" % len(img_names))
	count = 1
	for fn in img_names:
		log.info('processing %s... ' % fn)
		output_folder = fn+"_output"
		img = cv2.imread(fn, 0)

		if img is None:
			log.warning("Failed to load", fn)
			continue

		final_image_path = os.path.abspath(output_folder+"/cropped.jpg")

		file_exists = os.path.isfile(final_image_path)
		reference_file_exists = os.path.isfile(output_folder+"/"+"reference_stripe.pkl")
		# if not file_exists:
			# iu.undistort(fn)
		try:
			reference_info = {}
			if not file_exists:
				faceNormalizer.normalize(fn)
			if not reference_file_exists:
				reference_info, edged, striped = detectReferenceStripe.detect(final_image_path)	
				cv2.imwrite(output_folder+"/edged.jpg", edged)
				cv2.imwrite(output_folder+"/stripe.jpg", striped)
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
		except ZeroDivisionError as zeroError:
			log.exception("Exception processing reference stripe in %s", fn)
			continue

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

				# distances_few_eu = calculateDistancesCm.few(final_image_path, "img_"+str(count), points_dict, reference_info)
				# log.info("Aggreagating distances: FEW")
				# few_distances_dict_eu = aggregateDistances(distances_few_eu, few_distances_dict_eu, cf.OUTPUT_DIR+"/"+"distances_few", count, fn)
				
				distances_all_eu, distances_all_mh = calculateDistancesCm.all(points_dict, reference_info)
				log.info("Aggreagating distances cm: ALL")
				all_distances_dict_eu, all_distances_dict_mh = aggregateDistances(distances_all_eu, all_distances_dict_eu, 
																					distances_all_mh, all_distances_dict_mh, 
																					cf.OUTPUT_DIR + "/"+"distances_all_cm", count, fn)

				distances_farkas_eu, distances_farkas_mh = calculateDistancesCm.farkas(points_dict, reference_info)
				log.info("Aggreagating distances cm: FARKAS")
				farkas_distances_dict_eu, farkas_distances_dict_mh = aggregateDistances(distances_farkas_eu, farkas_distances_dict_eu, 
																						distances_farkas_mh, farkas_distances_dict_mh, 
																						cf.OUTPUT_DIR+"/"+"distances_farkas_cm", count, fn)

				distances_all_eu_px, distances_all_mh_px = calculateDistancesPx.all(points_dict, reference_info)
				log.info("Aggreagating distances px: ALL")
				all_distances_dict_eu_px, all_distances_dict_mh_px = aggregateDistances(distances_all_eu_px, all_distances_dict_eu_px, 
																					distances_all_mh_px, all_distances_dict_mh_px, 
																					cf.OUTPUT_DIR + "/"+"distances_all_px", count, fn)

				distances_farkas_eu_px, distances_farkas_mh_px = calculateDistancesPx.farkas(points_dict, reference_info)
				log.info("Aggreagating distances px: FARKAS")
				farkas_distances_dict_eu_px, farkas_distances_dict_mh_px = aggregateDistances(distances_farkas_eu_px, farkas_distances_dict_eu_px, 
																						distances_farkas_mh_px, farkas_distances_dict_mh_px, 
																						cf.OUTPUT_DIR+"/"+"distances_farkas_px", count, fn)

				distances_all_eu_px_1000 = copy.deepcopy(distances_all_eu_px)
				distances_all_eu_px_1000.update((x, y*1000) for x, y in distances_all_eu_px_1000.items())

				distances_all_mh_px_1000 = copy.deepcopy(distances_all_mh_px)
				distances_all_mh_px_1000.update((x, y*1000) for x, y in distances_all_mh_px_1000.items())
				log.info("Aggreagating distances px1000: ALL")

				all_distances_dict_eu_px_1000, all_distances_dict_mh_px_1000 = aggregateDistances(distances_all_eu_px_1000, all_distances_dict_eu_px_1000, 
																					distances_all_mh_px_1000, all_distances_dict_mh_px_1000, 
																					cf.OUTPUT_DIR + "/"+"distances_all_px_1000", count, fn)

				distances_farkas_eu_px_1000 = copy.deepcopy(distances_farkas_eu_px)
				distances_farkas_eu_px_1000.update((x, y*1000) for x, y in distances_farkas_eu_px_1000.items())

				distances_farkas_mh_px_1000 = copy.deepcopy(distances_farkas_mh_px)
				distances_farkas_mh_px_1000.update((x, y*1000) for x, y in distances_farkas_mh_px_1000.items())
				log.info("Aggreagating distances px1000: FARKAS")

				farkas_distances_dict_eu_px_1000, farkas_distances_dict_mh_px_1000 = aggregateDistances(distances_farkas_eu_px_1000, farkas_distances_dict_eu_px_1000, 
																					distances_farkas_mh_px_1000, farkas_distances_dict_mh_px_1000, 
																					cf.OUTPUT_DIR + "/"+"distances_farkas_px_1000", count, fn)

			count = count + 1 
			img_names_processed.append(fn)
		else:
			log.critical("Landmarking extraction did not went well! Finishing image processing...")
			continue
		os.chdir(cf.ROOT_DIR)
	log.info("Total: %d " % len(img_names_processed))	
	log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))
