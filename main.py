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

'''
	This is the main script that try to obtain facial distances of a 
	photo taken using the protocol defined for Tuany Dias Pinheiro.

	This is the pipeline:

	1 - Camera calibration (not yet implemented)
	2 - Photo normalizer (rotation)
	3 - Reference stripe detection (obtain reference coordinates and size)
	4 - Points extraction
	5 - Distances calculus
'''

if __name__ == '__main__':
	start_time = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image-folder", required=True,
		help="path to the input images folder")
	args = vars(ap.parse_args())
	
	root_path = cf.ROOT_DIR

	iu.undistort(args["image_folder"])

	# faceNormalizer.normalize( { "image_folder": args["image_folder"] } )
	final_image_path = root_path+"/"+args["image_folder"]+"/rotated.jpg"
	reference_stripe_args = { "image": final_image_path, "width": cf.REFERENCE_STRIPE_WIDTH }
	
	# reference_info is a dict: { "w-pixels": dB, 
	#                     "w-centimeters": dimB, 
	# 					  "h-pixels": dA, 
	#    				  "h-centimeters": dimA, 
	# 					  "pixelsPerMetric": pixelsPerMetric, 
	# 					  "coordinates": [tl, tr, br, bl] }
	reference_info = detectReferenceStripe.detect(reference_stripe_args)

	# os.chdir(args["image_folder"])

	# print("Saving reference_stripe results")
	# try:
	# 	csvfile = open(root_path+"/"+args["image_folder"]+"/"+"reference_stripe.csv", 'w')
	# 	fieldnames = ['w-pixels', 'w-centimeters', 'h-pixels', 'h-centimeters', 'pixelsPerMetric', 'coordinates']
	# 	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	# 	writer.writeheader()
	# 	writer.writerow(reference_info)
	# except IOError as e:
	#     if e.errno == errno.EACCES:
	#         print("--No write permittion")
	#         os.chdir("../")
	#     # Not a permission error.
	#     raise

	new_image_folder = root_path+"/"+args["image_folder"]
	os.chdir(root_path+"/"+"exec/OpenFace")
	process_out = subprocess.check_call(["FeatureExtraction", "-f", final_image_path, "-of", new_image_folder+"/points.csv", "-oi", new_image_folder+"/marked.jpg", "-no3Dfp", "-noMparams", "-noPose", "-noAUs", "-noGaze"], shell=True)
	process_out = 0

	if process_out == 0:
		points_dict = {}
		try:
			points_file = open(new_image_folder+"/"+"points.csv", 'rb')
			reader = csv.reader(points_file)
			headers = reader.next()
			for row in reader:
				for h, v in zip(headers, row):
					points_dict[h.strip()] = float(v)	
		finally:
			points_file.close()
		distances = calculateDistances.farkas(points_dict, reference_info)
		csvfile = open(root_path+"/"+args["image_folder"]+"/"+"distances.csv", 'w')
		writer = csv.DictWriter(csvfile, fieldnames=distances.keys())
		writer.writeheader()
		writer.writerow(distances)
	else:
		raise Exception('Landmarking extraction did not went well! Finishing image processing...')
	print("--- Total execution time: %s seconds ---" % (time.time() - start_time))