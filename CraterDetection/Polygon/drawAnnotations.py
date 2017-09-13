#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

import numpy as np
import json
import glob, os
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

rootDir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_NAC/'
annotationDir = rootDir+'annotations/'
uniqueBoxes = []
num_craters = 0

pos_file_names = glob.glob(annotationDir+'*_VOC.json')
for filename in pos_file_names:
    data = json.load(open(filename))

    image_name = filename.split(annotationDir)[1].split('_VOC.json')[0]
    output_name = 'annotated_' + image_name
    image_filename = (annotationDir + image_name + '.tif')

    #format = "GTiff"
    #driver = gdal.GetDriverByName(format)
    #dst_ds = driver.CreateCopy(image_filename, gdal.Open(imageDir + image_name + '.IMG'), 0,
    #                           ['TILED=YES', 'COMPRESS=PACKBITS'])


    output_filename = (annotationDir + output_name + '.tif')

    if (not os.path.isfile(image_filename)):
        print('Image not found: ' + image_filename)
    else:
        ds = gdal.Open(image_filename)
        image = np.array(ds.GetRasterBand(1).ReadAsArray())

        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gray')

        rgb_image = np.dstack((np.dstack((image, image)), image)) #3 channel

        for bbox in data['object']:
            box = bbox['bndbox']

            ax.add_patch(
                patches.Rectangle(
                    [int(box['xmin']) - 1, int(box['ymin']) - 1],
                    (int(box['xmax']) - int(box['xmin'])),
                    (int(box['ymax']) - int(box['ymin'])),  # (x,y), width, height
                    fill=False,
                    ec='g',
                )
            )

            for y in range(int(box['ymin'])-1, int(box['ymax'])+1):
                rgb_image[y, int(box['xmin']) - 1: int(box['xmax'])] = [0, 255, 0]

            #rgb_image[int(box['ymin']) - 2, int(box['xmin']) - 2: int(box['xmax']) + 1, :] = [0, 255, 0] #ymin line
            #rgb_image[int(box['ymin']) - 1, int(box['xmin']) - 1: int(box['xmax']), :] = [0, 255, 0] #ymin line
            #rgb_image[int(box['ymin']), int(box['xmin']): int(box['xmax'])-1, :] = [0, 255, 0] #ymin line


            #rgb_image[int(box['ymax']) -1, int(box['xmin']): int(box['xmax']) -1 , :] =  [0, 255, 0] #ymax line
            #rgb_image[int(box['ymax']), int(box['xmin']) - 1: int(box['xmax']), :] = [0, 255, 0]  # ymax line
            #rgb_image[int(box['ymax']) +1, int(box['xmin']) - 2: int(box['xmax']) +1, :] = [0, 255, 0]  # ymax line


            #rgb_image[int(box['ymin']) - 2: int(box['ymax']) +1, int(box['xmin']) - 2, :] =  [0, 255, 0] #xmin line
            #rgb_image[int(box['ymin']) - 1: int(box['ymax']), int(box['xmin']) - 1, :] =  [0, 255, 0] #xmin line
            #rgb_image[int(box['ymin']): int(box['ymax']) -1, int(box['xmin']), :] =  [0, 255, 0] #xmin line

            #rgb_image[int(box['ymin']) - 2: int(box['ymax']) +1, int(box['xmax']) +1, :] =  [0, 255, 0] #xmax line
            #rgb_image[int(box['ymin']) - 1: int(box['ymax']), int(box['xmax']), :] = [0, 255, 0]  # xmax line
            #rgb_image[int(box['ymin']): int(box['ymax']) -1, int(box['xmax']) -1, :] = [0, 255, 0]  # xmax line

        image = Image.fromarray(rgb_image)
        if (image != None):
            image.save(output_filename)