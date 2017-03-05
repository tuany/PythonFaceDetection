# PythonFaceDetection

faceDetector.py is a simple face detection script using Haar Cascades Classifiers based on template available on: http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html

faceNormalizer.py is a script to normalize geometrically facial images. The face is detected in the image then the algorithm try to find an angle alpha to rotate and save the rotated final image.

detectReferenceStripe.py detects the top most object of the image as reference to estimate the pixel distances in centimeters. The reference object need to be retangular

Requires OpenCV 3.0.0, Pillow and OpenFace
