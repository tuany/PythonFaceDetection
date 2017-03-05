import argparse
import detectReferenceStripe
import faceNormalizer

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image-folder", required=True,
		help="path to the input images folder")
	ap.add_argument("-w", "--width", type=float, required=True,
		help="width of the reference stripe object in the image (in centimeters)")
	args = vars(ap.parse_args())
	
	faceNormalizer.normalize( { "image_folder": args["image_folder"] } )

	reference_stripe_args = { "image": args["image_folder"]+"/rotated.jpg", "width": args["width"] }
	result = detectReferenceStripe.detect(reference_stripe_args)
	print(result)
