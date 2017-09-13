#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

#This script is used to split the LOLA_DEM South Pole Large Tiles into smaller tiles for ingestion.

import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from osgeo import gdal

full_size = [30400, 30400]
p_size = [3800, 3800]
cut_size = [32,32]
stride = np.divide(cut_size, 2)

thisDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(thisDir, os.pardir, os.pardir)
dataDir = os.path.join(rootDir, 'Data')
DEMDir = os.path.join(dataDir, 'LOLA_DEM', 'South_Pole')
DEMLargeDir = os.path.join(DEMDir, 'Large_Tiles')
DEMSmallDir = os.path.join(DEMDir, 'Small_Tiles')

base_filename = "hs-45-45_lola20sp"
#https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio

v_pieces = np.floor_divide(full_size[0], p_size[0]) #Number of vertical divisions for large tiles
h_pieces = np.floor_divide(full_size[1], p_size[1]) #Number of horizontal divisions for large tiles

for n in (25,39,43,57,64):
    if not os.path.isdir(os.path.join(DEMSmallDir,'P'+str(n+1),'')):
        os.mkdir(os.path.join(DEMSmallDir,'P'+str(n+1),''))

    curr_filename = os.path.join(DEMLargeDir,base_filename+'_p'+str(n+1)+'.tif')
    ds = gdal.Open(curr_filename)
    im = np.array(ds.GetRasterBand(1).ReadAsArray())
    width = im.shape[1]
    height = im.shape[0]


    y_ind, x_ind = np.floor_divide(n, v_pieces), np.mod(n, v_pieces)
    y_pos, x_pos = [0] * 2, [0] * 2
    y_pos[0], x_pos[0] = np.multiply(p_size[0], y_ind), np.multiply(p_size[1], x_ind)
    y_pos[1], x_pos[1] = y_pos[0] + p_size[0], x_pos[0] + p_size[1]
    h_cuts = np.floor_divide(p_size[1], stride[1]) - (np.floor_divide(cut_size[1], stride[1])) + 1
    v_cuts = np.floor_divide(p_size[0], stride[0]) - (np.floor_divide(cut_size[0], stride[0])) + 1

    #The below is what was used to generate the tiles found in this github, however they are incorrect.
    #The correct formula is given above.
    #Once the data provided has been refactored, the below formula will be replaced.
    w_cuts = np.multiply(np.floor_divide(width, cut_size[1]), np.divide(cut_size[1], stride[1]))
    h_cuts = np.multiply(np.floor_divide(height, cut_size[0]), np.divide(cut_size[0], stride[0]))
    for i in range(w_cuts+1):
        for j in range(h_cuts+1):
            x_off = np.multiply(i, stride[1])
            y_off = np.multiply(j, stride[0])
            #image = np.asarray(im)
            image = im[y_off:y_off+cut_size[0], x_off:x_off+cut_size[1]]
            ind = (i*w_cuts + j)

            #x = i*cut_size[1]+x_pos[0]
            #y = j*cut_size[0]+y_pos[0]
            #filename = os.path.join(DEMSmallDir,'P'+str(n+1),base_filename+'_x'+str(x)+'_y'+str(y))
            # Once existing data names have been refactored, the below filename will be replaced with the above.
            filename = os.path.join(DEMSmallDir,'P'+str(n+1),base_filename+'_cut'+str(ind))
            im2 = Image.fromarray(image)
            im2.save(filename + '.tif')


