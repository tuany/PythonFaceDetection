import argparse
import detectReferenceStripe
import faceNormalizer
import calculateDistances
import csv
import os
import subprocess

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
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image-folder", required=True,
		help="path to the input images folder")
	ap.add_argument("-w", "--width", type=float, required=True,
		help="width of the reference stripe object in the image (in centimeters)")
	args = vars(ap.parse_args())
	
	root_path = os.getcwd()

	faceNormalizer.normalize( { "image_folder": args["image_folder"] } )
	final_image_path = root_path+"/"+args["image_folder"]+"/rotated.jpg"
	reference_stripe_args = { "image": final_image_path, "width": args["width"], "s_width": 0.617, "s_height": 0.455, "focal_length": 0.4 }
	
	# reference_info is a dict: { "w-pixels": dB, 
	#                     "w-centimeters": dimB, 
	# 					  "h-pixels": dA, 
	#    				  "h-centimeters": dimA, 
	# 					  "pixelsPerMetric": pixelsPerMetric, 
	# 					  "coordinates": [tl, tr, br, bl] }
	reference_info = detectReferenceStripe.detect(reference_stripe_args)

	os.chdir(args["image_folder"])

	print("Saving reference_stripe results")
	try:
		csvfile = open(root_path+"/"+args["image_folder"]+"reference_stripe.csv", 'w')
		fieldnames = ['w-pixels', 'w-centimeters', 'h-pixels', 'h-centimeters', 'pixelsPerMetric', 'coordinates']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow(reference_info)
	except IOError as e:
	    if e.errno == errno.EACCES:
	        print("--No write permittion")
	        os.chdir("../")
	    # Not a permission error.
	    raise

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
		distances = calculateDistances.distances(points_dict, reference_info)
	else:
		raise Exception('Landmarking extraction did not went well! Finishing image processing...')