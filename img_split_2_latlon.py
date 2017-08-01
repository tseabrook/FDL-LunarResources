from pathlib import Path
import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from osgeo import gdal
from testNAC import long_id, lat_id

base_folder = "/Volumes/DATA DISK/PDS_FILES/LROC_BigDEM/"

folder_switch = {
    0 : 'crater',
    1 : 'not crater',
    2 : 'not sure',
}

base_filename = "hs-45-45_lola20sp"

full_size = [30400, 30400]
p_size = [3800, 3800]
cut_size = [32, 32]
stride = np.divide(cut_size,2)

for n in (26):
    v_pieces = np.floor_divide(full_size[0], p_size[0])
    h_pieces = np.floor_divide(full_size[1], p_size[1])
    y_ind = np.floor_divide(n, v_pieces)
    x_ind = np.mod(n, v_pieces)
    y_pos = [0]*2
    x_pos = [0]*2
    y_pos[0] = np.multiply(p_size[0],y_ind)
    x_pos[0] = np.multiply(p_size[1],x_ind)
    y_pos[1] = y_pos[0] + p_size[0]
    x_pos[1] = x_pos[0] + p_size[1]




for m in len(folder_switch):
    folder_name = folder_switch[m]
    if not os.path.isfile(base_folder + folder_name):
        os.mkdir(base_folder + folder_name)
    for filename in os.listdir(base_folder + folder_name):
        piece_id = filename.split('_cut')[1].split('.')[0]

    w_cuts = np.multiply(np.floor_divide(p_size[1], cut_size[1]), np.divide(cut_size[1], stride[1]))
    h_cuts = np.multiply(np.floor_divide(p_size[0], cut_size[0]), np.divide(cut_size[0], stride[0]))

    y_ind = np.floor_divide(piece_id, w_cuts)
    x_ind = np.mod(piece_id, w_cuts)

    y_pos = np.multiply(y_ind, stride[0])
    x_pos = np.multiply(x_ind, stride[1])

#https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
for n in (25,39,43,57,64):
    if not os.path.isdir(base_folder+'p'+str(n+1)+"/"):
        os.mkdir(base_folder+'p'+str(n+1)+"/")
    curr_filename = base_folder+base_filename+'_p'+str(n+1)+'.tif'
    ds = gdal.Open(curr_filename)
    im = np.array(ds.GetRasterBand(1).ReadAsArray())
    width = im.shape[1]
    height = im.shape[0]
    w_cuts = np.multiply(np.floor_divide(width, output_size[1]), np.divide(output_size[1], stride[1]))
    h_cuts = np.multiply(np.floor_divide(height, output_size[0]), np.divide(output_size[0], stride[0]))
    for i in range(w_cuts):
        for j in range(h_cuts):
            x_off = np.multiply(i, stride[1])
            y_off = np.multiply(j, stride[0])
            #image = np.asarray(im)
            image = im[y_off:y_off+output_size[0], x_off:x_off+output_size[1]]
            ind = (i*w_cuts + j)
            filename = base_folder+'p'+str(n+1)+"/"+base_filename+'_cut'+str(ind)
            im2 = Image.fromarray(image)
            im2.save(filename + '.tif')