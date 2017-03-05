import argparse
import detectReferenceStripe
import faceNormalizer
import csv
import os
import subprocess

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image-folder", required=True,
		help="path to the input images folder")
	ap.add_argument("-w", "--width", type=float, required=True,
		help="width of the reference stripe object in the image (in centimeters)")
	args = vars(ap.parse_args())
	
	faceNormalizer.normalize( { "image_folder": args["image_folder"] } )
	final_image_path = args["image_folder"]+"/rotated.jpg"
	reference_stripe_args = { "image": final_image_path, "width": args["width"] }
	result = detectReferenceStripe.detect(reference_stripe_args)

	# os.chdir(args["image_folder"])

	# print("Saving reference_stripe results")
	# try:
	# 	csvfile = open("reference_stripe.csv", 'w')
	# 	fieldnames = ['pixels', 'centimeters', 'coordinates']

	# 	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	# 	writer.writeheader()
	# 	writer.writerow(result)
	# except IOError as e:
	#     if e.errno == errno.EACCES:
	#         print("--No write permittion")
	#         os.chdir("../")
	#     # Not a permission error.
	#     raise

	new_final_image_path = "../../"+final_image_path
	new_image_folder = "../../"+args["image_folder"]
	os.chdir("exec/OpenFace")
	process_out = subprocess.check_call(["FaceLandmarkImg.exe", "-f", new_final_image_path, "-of", new_image_folder+"/points.csv", "-oi", new_image_folder+"/marked.jpg"], shell=True)
	# process_out = 1

	if process_out == 0:
		os.chdir("../../")
		print("Calculating distances:")
	else:
		raise Exception('Landmarking extraction did not went well! Finishing image processing...')