from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse
import os
# from imutils import contours as ct

def PIL2array(img):
    return np.array(img, np.uint8)

current_dir = os.getcwd()

image = Image.open(current_dir+"/img/DSCN3451AlbertPinheiroBarboza.JPG").convert(mode='L')
image.thumbnail((450,450), Image.ANTIALIAS)

image = PIL2array(image)
print(image.shape)
# coords = corner_peaks(corner_harris(image, k=0.00000000000001), min_distance=5)
# coords_subpix = corner_subpix(image, coords, window_size=13)

# fig, ax = plt.subplots()
# ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
# ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
# ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
# ax.axis((0, 450, 450, 0))
# plt.show()


edges1 = feature.canny(image)
edges2 = feature.canny(image, sigma=2)

# display results
fig, (ax2, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(8, 2),
                                    sharex=True, sharey=True)

# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=2$', fontsize=20)

fig.tight_layout()

plt.show()