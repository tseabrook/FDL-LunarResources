from PIL import Image
import glob, os
import numpy as np
from osgeo import gdal

#imageDir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_NAC/annotations/'
#imageDir = '/Volumes/DATA DISK/PDS_FILES/LROC_NAC/Repo/imgs/LROC_NAC/'
imageDir = '/home/tseabrook/NAC_Images/'
outDir = imageDir + 'Resampled/'
sampleFactor = 40
uniqueBoxes = []
num_craters = 0

gdal.UseExceptions()

def filterBadImage(image):
    perc = np.divide(np.sum(int(image <= 5)), (np.sum(image.size)))
    if(perc > 0.95):
        image = None
    return image

pos_file_names = glob.glob(imageDir+'*.tif')
for filename in pos_file_names:

    output_filename = outDir + filename.split('.')[0].split(imageDir)[1] + '_d' + str(sampleFactor) + '.tif'

    ds = gdal.Open(filename)
    image = ds.GetRasterBand(1).ReadAsArray()
    if(image is not None):
        image = np.array(image)
        #image = Image.open(filename).load()
        height, width = image.shape
        y_over = np.mod(height, sampleFactor)
        x_over = np.mod(width, sampleFactor)
        image = image[0:-y_over, 0:-x_over]
        y = np.floor_divide(height, sampleFactor)
        x = np.floor_divide(width, sampleFactor)

        image = Image.fromarray(image)
        image = image.resize((x, y), Image.ANTIALIAS)
        #image = filterBadImage(image)

        if(image != None):
            image.save(output_filename)