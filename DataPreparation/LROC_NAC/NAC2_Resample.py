from PIL import Image
import glob, os
import numpy as np
from osgeo import gdal

#resampleImage
#Purpose:
#Resamples NAC images to match the resolution found in the LOLA_DEM, so that feature matching can take place.

thisDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(thisDir, os.pardir, os.pardir)
dataDir = os.path.join(rootDir, 'Data')
NACDir = os.path.join(dataDir, 'LROC_NAC', 'South_Pole', 'Images')
outDir = os.path.join(dataDir, 'LROC_NAC', 'South_Pole', 'Resampled')

if(not os.path.isdir(NACDir)): #SUBJECT TO RACE CONDITION
    print('No Images found, please first execute script: \'NAC1_PDS_GRAB.py\'')

if(not os.path.isdir(outDir)): #SUBJECT TO RACE CONDITION
    os.makedirs(outDir)

sampleFactor = 40 #Factor of 40 resamples NAC from 0.5m to 20m resolution (same as DEM)

gdal.UseExceptions()

def filterBadImage(image, threshold=0.95):
    #NOT USED CURRENTLY,
    # filters images with darkness percentage more than threshold
    perc = np.divide(np.sum(int(image <= 5)), (np.sum(image.size)))
    if(perc > threshold):
        image = None
    return image

def resample_image(image, sampleFactor):
    image = np.array(image)
    # image = Image.open(filename).load()
    height, width = image.shape
    y_over = np.mod(height, sampleFactor)
    x_over = np.mod(width, sampleFactor)
    #Trims excess pixels prior to resampling
    if(y_over > 0):
        image = image[0:-y_over,:]
    if(x_over > 0):
        image = image[:,0:-x_over]
    y = np.floor_divide(height, sampleFactor)
    x = np.floor_divide(width, sampleFactor)
    image = Image.fromarray(image)
    #ANTIALIAS method used
    image = image.resize((x, y), Image.ANTIALIAS)
    return image

pos_file_names = glob.glob(os.path.join(NACDir,'*.tif'))
for filename in pos_file_names:

    output_filename = outDir + filename.split(NACDir)[1].split('.')[0] + '_d' + str(sampleFactor) + '.tif'

    ds = gdal.Open(filename)
    image = ds.GetRasterBand(1).ReadAsArray()
    if(image is not None):
        image = resample_image(image, sampleFactor)

        if(image is not None):
            image.save(output_filename)