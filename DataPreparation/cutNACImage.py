from PIL import Image
import glob, os
import numpy as np
from osgeo import gdal

#imageDir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_NAC/annotations/'

gdal.UseExceptions()

output_size = [32,32]
stride = np.divide(output_size, 2)
downsample_ratio = 40
imageDir = '/Volumes/DATA DISK/PDS_FILES/LROC_NAC/LRO_New/Resampled - S3Server/'
#resampledDir = imageDir + 'Resampled/'
#imageDir = '/home/tseabrook/NAC_Images/'

pos_file_names = glob.glob(imageDir+'*.tif')
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
                    #image = np.asarray(im)
                    image = im[y_off:y_off+output_size[0], x_off:x_off+output_size[1]]
                    perc_dark = np.divide(np.sum((image < 10).astype(np.uint8)), (np.sum(image.size)))
                    if(perc_dark < 0.4): #Tile more than 40% illuminated
                        ind = ((i*v_cuts) + j)

                        name, ext = filename.split('.')
                        #name, downsample_ratio = name.split('_div')
                        name = name.split(imageDir)[1]

                        output_filename = imageDir + 'Tiles/' + name \
                            + '_x' + str(x_off) + '_y' + str(y_off) + '.' + ext
                        im2 = Image.fromarray(image)
                        im2.save(output_filename)