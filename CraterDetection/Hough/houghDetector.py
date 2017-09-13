#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

#This script is really slow for large images, so should only be executed on those resampled.
#This is a first baseline circle detector to identify the limitations of convential techniques for crater detection.

import numpy as np
import matplotlib.pyplot as plt
import os, glob
from osgeo import gdal

from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.feature import canny

thisDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(thisDir, os.pardir, os.pardir)
dataDir = os.path.join(rootDir, 'Data')
NACDir = os.path.join(dataDir, 'LROC_NAC', 'South_Pole', 'Resampled')

pos_file_names = glob.glob(os.path.join(NACDir,'*.tif'))

for i in range (np.minimum(pos_file_names.__len__, 10)):
    filename = pos_file_names[i]
    # Load picture and detect edges
    ds = gdal.Open(filename)
    image = ds.GetRasterBand(1).ReadAsArray()
    if(image is not None):
        # Low threshold and High threshold represent number of pixels that may be skipped and max consisting pixels to make a line [4, 60 seems good]
        # Sigma represents the width of the gaussian smoothing kernel [3 seems good]
        edges = canny(image, sigma=3, low_threshold=4, high_threshold=60)

        # Detect two radii
        hough_radii = np.arange(20, 100, 2)
        hough_res = hough_circle(edges, hough_radii)

        # Select the most prominent circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=10)

        # Draw them
        fig, axarr = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
        image = color.gray2rgb(image)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius)
            image[circy, circx] = (220, 20, 20)

        axarr[1].imshow(image, cmap=plt.cm.gray)
        axarr[0].imshow(edges, cmap=plt.cm.gray)
        plt.show()