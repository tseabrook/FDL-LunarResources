import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from osgeo import gdal

output_size = [32,32]
stride = np.divide(output_size, 2)
base_folder = "/Volumes/DATA DISK/PDS_FILES/LROC_BigDEM/"
base_filename = "hs-45-45_lola20sp"
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

#folder_path = '/PDS_IMAGES/LROC_DEM'
#
#names = [os.path.basename(file) for file in glob.glob(folder_path)]
#for file in glob.glob(folder_path+"/*.tif"):
#    im = Image.open(file)
#    cover = im.resize((256, 256), Image.ANTIALIAS)
#    filename = os.path.basename(file).split('.')
#    im.save(filename[0]+'.jpeg')

