import numpy as np
import matplotlib.pyplot as plt

from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter


filename = 'imgs/testdata/m108898482_cdr_jp2_p'

for i in range(64):
    curr_filename = filename+str(i+1)+'.jp2'


    # Load picture and detect edges
    image = glymur.Jp2k(curr_filename)[:]
    # Low threshold and High threshold represent number of pixels that may be skipped to make a line [4, 60 seems good]
    # Sigma represents the width of the guassian smoothing kernel [3 seems good]
    edges = canny(image, sigma=3, low_threshold=4, high_threshold=60)
    fig, axarr = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    #axarr[1].imshow(image, cmap=plt.cm.gray)
    #plt.show()

    # Detect two radii
    hough_radii = np.arange(20, 100, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=10)

    # Draw them
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius)
        image[circy, circx] = (220, 20, 20)

    axarr[1].imshow(image, cmap=plt.cm.gray)
    axarr[0].imshow(edges, cmap=plt.cm.gray)
    plt.show()