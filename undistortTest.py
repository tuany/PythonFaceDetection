from __future__ import print_function
from glob import glob
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
import config as cf
import logger

start_time = time.time()
log = logger.getLogger(__file__)
log.info("Image processing started")
log.info("Switching to dir %s " % cf.ROOT_DIR)
os.chdir(cf.ROOT_DIR)

# get all images in img/ directory 
img_names = glob(cf.INPUT_IMG_MASK)

log.info("%4d images found!" % len(img_names))
count = 1
for fn in img_names:
	log.info('processing %s... ' % fn)
	output_folder = fn+"_output"
	img = cv2.imread(fn, 0)

	if img is None:
		log.warning("Failed to load", fn)
		continue

	final_image_path = os.path.abspath(output_folder+"/undistorted.jpg")

	file_exists = os.path.isfile(final_image_path)
	reference_file_exists = False#os.path.isfile(output_folder+"/"+"reference_stripe.pkl")
	# if not file_exists:
	iu.undistort(fn)
	try:
		reference_info = {}
		if not file_exists:
			faceNormalizer.normalize(fn)
		if not reference_file_exists:
			reference_info, edged, striped = detectReferenceStripe.detect(final_image_path)	
			cv2.imwrite(output_folder+"/edged2.jpg", edged)
			cv2.imwrite(output_folder+"/stripe2.jpg", striped)
			try:
				reference_stripe_file = output_folder+"/"+"reference_stripe2"
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
			with open(output_folder+"/"+"reference_stripe2.pkl", "rb") as f:
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
	os.chdir(cf.ROOT_DIR)
log.info("Total: %d " % len(img_names_processed))	
log.info("--- Total execution time: %s minutes ---" % ((time.time() - start_time)/60))