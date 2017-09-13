#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

from PIL import Image
import glob, os
import numpy as np
from osgeo import gdal

#imageDir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_NAC/annotations/'

gdal.UseExceptions()

output_size = [32,32]
stride = np.divide(output_size, 2)

thisDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(thisDir, os.pardir, os.pardir)
dataDir = os.path.join(rootDir, 'Data')
NACDir = os.path.join(dataDir, 'LROC_NAC', 'South_Pole', 'Resampled')
TileDir = os.path.join(dataDir, 'LROC_NAC', 'South_Pole', 'Tiles')

if(not os.path.isdir(NACDir)): #SUBJECT TO RACE CONDITION
    print('No Images found, please first execute script: \'NAC2_Resample.py\'')

if(not os.path.isdir(TileDir)): #SUBJECT TO RACE CONDITION
    os.makedirs(TileDir)

#resampledDir = imageDir + 'Resampled/'
#imageDir = '/home/tseabrook/NAC_Images/'

pos_file_names = glob.glob(os.path.join(NACDir,'*.tif'))
for filename in pos_file_names:
    ds = gdal.Open(filename)
    im = ds.GetRasterBand(1).ReadAsArray()
    if(im is not None):
        perc_dark = np.divide(np.sum((im < 10).astype(np.uint8)), (np.sum(im.size)))
        if(perc_dark <= 0.95): #Image more than 5% illuminated
            width,height = im.shape[1], im.shape[0]

            #old(wrong)    h_cuts = np.multiply(np.floor_divide(width, output_size[1]), np.divide(output_size[1], stride[1])) #horizontal cuts
            h_cuts = np.floor_divide(width, stride[1]) - (np.floor_divide(output_size[1], stride[1])) + 1
            #old(wrong)    v_cuts = np.multiply(np.floor_divide(height, output_size[0]), np.divide(output_size[0], stride[0])) #vertical cuts
            v_cuts = np.floor_divide(height, stride[0]) - (np.floor_divide(output_size[0], stride[0])) + 1
            for i in range(h_cuts):
                for j in range(v_cuts):
                    x_off = np.multiply(i, stride[1]) #tile top-left x position
                    y_off = np.multiply(j, stride[0]) #tile top-left y position
                    tile = im[y_off:y_off+output_size[0], x_off:x_off+output_size[1]]
                    perc_dark = np.divide(np.sum((tile < 10).astype(np.uint8)), (np.sum(tile.size)))
                    if(perc_dark < 0.4): #Tile more than 40% illuminated
                        ind = ((i*v_cuts) + j)

                        name = filename.split(os.path.join(NACDir, ''))[1]
                        name, ext = name.split('.')

                        output_filename = os.path.join(TileDir, name \
                            + '_x' + str(x_off) + '_y' + str(y_off) + '.' + ext)
                        if (not os.path.isfile(output_filename)):
                            tile = Image.fromarray(tile)
                            tile.save(output_filename)