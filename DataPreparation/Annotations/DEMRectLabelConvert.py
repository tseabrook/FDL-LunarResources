#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from osgeo import gdal
import json
from pprint import pprint
import xml.etree.ElementTree as ET
import matplotlib.patches as patches

#The purpose of this file is to stitch together images that belong to the same map.
#These images have previously been cut up to make annotating those images an easier task.

#The images were mostly annotated using RectLabel, this outputs the annotations into a JSON format.
#For our labels to be compatible with the NEON framework provided by Intel-Nervana,
#   we need to convert our JSON labels into PASCAL VOC XML format used by ImageNet
#
# USEFUL LINKS :
#   Python Misc Functions (user contribution)
#       https://github.com/mprat/pascal-voc-python/blob/master/voc_utils/voc_utils.py
#       This uses a BeautifulSoup data structure for loading xmls.
#   LabelImg Git
#       https://github.com/tzutalin/labelImg
#       This is used by Intel-Nervana employees to annotate images with bounding boxes.
#
#
# PASCAL VOC description:
# (From ImageNet Attribute Labels README: http://image-net.org/downloads/attributes/README)

# BBOX
#   Each bounding box contains the fields x1, x2, y1, y2, all normalized to be
#   between 0 and 1.
#
# Sample code from LabelImg/tests
#   from pascal_voc_io import PascalVocWriter
#   from pascal_voc_io import PascalVocReader
#   writer = PascalVocWriter('tests', 'test', (512, 512, 1), localImgPath='tests/test.bmp')
#   difficult = 1
#   writer.addBndBox(60, 40, 430, 504, 'person', difficult)
#   writer.addBndBox(113, 40, 450, 403, 'face', difficult)
#   writer.save('tests/test.xml')
#   reader = PascalVocReader('tests/test.xml')
#   shapes = reader.getShapes()
#
# A description of the problem is given:
#
# The LOLA South-Pole DEM has previously been split into 8x8 tiles using img_split.m.
# The original DEM file has dimensions 30400x30400 which equates to a range of -304000m : +304000m at 20m resolution
#
# !!! As a result of splitting being performed in MATLAB,
#   the tiling sequence follows a left-to-right row-by-row ordering. !!!
#
# Each of the resulting 64 tiles contains 3800x3800 pixels or 76kmx76km.
# This was still a large area to attempt to label.
# The 26th 'Macro' tile was chosen as a model training set due to its diversity of landscape features.
# This tile was further split into 32px by 32px 'Micro' tiles to allow for quick classification
#
# The 26th Tile represents the tile in the 2nd column, 4th row of the LOLA DEM.
# Region (-228km E, 76km N) to (-152km E, 0km N)
#   The SouthPole LOLA DEM is in Equirectangular Projection.
#   As a result, the -85E, 85S to 85E, -85S is divisible by the distance
#   304000m / 5 = 60800m per degree
#   5 / 304000m = 0.000016447368421degrees per meter
#
# !!! 3800 is not directly divisible by 32, so some clipping has occurred for each 'Macro' Tile
#   An indexing error in the original code resulted in an additional
#   micro-tile being lost from the end of each row and column !!!
#
# The resulting 'Micro' Tiles cover an original region of 3744 x 3744 pixels
#   i.e. 74880m * 74880m
# Region (-228km E, 76km N) to (-153.12km E, 1.12km N)
#
# Each of 54756 32*32px tiles, covering 640m*640m follow a top-to-bottom, column by column tiling sequence.
# The img_split.py file was used to split these segments.
#
# Labelling has been performed for the first 18,000 tiles.
# Annotation has been performed for 3000 of the 18,000 tiles.
#   Namely ID 9000-12000
#   We don't need to restitch the images together.
#   However, we do need to restitch the annotations,
#     and reinstate properly coordinates

def tileID_to_x_y(id, width, height, order = 'TopBottom'):
    # Find indexes of cut piece depending on tiling order
    if (order == 'TopBottom'):
        x = np.floor_divide(id, height)
        y = np.mod(id, height)
    elif (order == 'LeftRight'):
        y = np.floor_divide(id, width)
        x = np.mod(id, width)
    else:
        print('expected order: TopBottom or LeftRight, assumed former.')
        x = np.floor_divide(id, height)
        y = np.mod(id, height)

    return x, y

def get_box_corners(x,y,width,height):
    xMin = x*(width/2) #Stride is a half
    xMax = xMin + width
    yMin = y*(height/2) #Stride is a half
    yMax = yMin + height
    return xMin, xMax, yMin, yMax

def RectLabel_2_PASCAL_VOC(object, corners, name, difficult = False):
    x, y, w, h = object['x_y_w_h']
    xMin, xMax, yMin, yMax = corners
    voc_annotation = {
        "bndbox": {
            "xmax": xMin + x + w,
            "xmin": xMin + x,
            "ymax": yMin + y + h,
            "ymin": yMin + y, },
        "difficult": difficult,
        "name": name,
    }
    return voc_annotation

def checkUniqueBox(bbox, uniqueBoxes):
    returnValue = []
    if(bbox not in uniqueBoxes):
         uniqueBoxes.append(bbox)
         returnValue.append(bbox)
    return returnValue

#Variables Definition:
xSplits = 236
ySplits = 236

order = 'TopBottom'
rootDir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_BigDEM/'
annotationDir = rootDir + 'annotations/'
xmlImgDir = rootDir + '/TimSort/6000-8999/Crater/'
uniqueBoxes = []
PASCALVOC_filename = annotationDir+'hs-45-45_lola20sp_p26_VOC.json'

ds = gdal.Open(rootDir + 'hs-45-45_lola20sp_p26.tif')
img = np.array(ds.GetRasterBand(1).ReadAsArray())

fig, ax = plt.subplots(1)
ax.imshow(img, cmap='gray')

if(os.path.isfile(PASCALVOC_filename)):
    output = json.load(open(PASCALVOC_filename))
    uniqueBoxes = output['object']
else:
    output = {'object': []}

pos_file_names = glob.glob(annotationDir+'*.xml')
for filename in pos_file_names:

    source_name, cut_id = filename.split('_cut')
    cut_id, ext = cut_id.split('.')

    source_name = source_name.split(annotationDir)[-1]
    source_img = xmlImgDir + source_name + '_cut' + cut_id + '.tif'

    cut_id = int(cut_id)  # convert ID from string

    tree = ET.parse(filename)
    root = tree.getroot()

    # Read annotation.json (Assume format of RectLabel output)
    xPix, yPix = int(root.find('size')._children[0].text), int(root.find('size')._children[1].text) # Width and height of original image
    xTileMax = ((xPix/2) * xSplits) + (xPix/2)
    yTileMax = ((yPix/2) * ySplits) + (yPix/2)

    # Convert TileID to x,y poss
    x_ind, y_ind = tileID_to_x_y(cut_id, xSplits, ySplits)

    # Find corner pixels of cut piece depending on index
    xMin, xMax, yMin, yMax = get_box_corners(x_ind, y_ind, xPix, yPix)

    for object in root.findall('object'):
        bndbox = object._children[4]

        #ds = gdal.Open(source_img)
        #img = np.array(ds.GetRasterBand(1).ReadAsArray())

        #fig, ax = plt.subplots(1)
        #ax.imshow(img, cmap='gray')
        #ax.add_patch(
        #    patches.Rectangle(
        #        [int(bndbox._children[0].text)-1, int(bndbox._children[1].text)-1], (int(bndbox._children[2].text) - int(bndbox._children[0].text)),
        #        (int(bndbox._children[3].text) - int(bndbox._children[1].text)),  # (x,y), width, height
        #        fill=False
        #    )
        #)

        voc_annotation = {
            "bndbox": {
                "xmax": int(bndbox._children[2].text)+1 + xMin,
                "xmin": int(bndbox._children[0].text)-1 + xMin,
                "ymax": int(bndbox._children[3].text)+1 + yMin,
                "ymin": int(bndbox._children[1].text)-1 + yMin,},
            "difficult": object._children[3].text,
            "name": object._children[0].text,
        }
        bbox = checkUniqueBox(voc_annotation, uniqueBoxes)
        if (len(bbox) > 0):
            output['object'].append(voc_annotation)

            ax.add_patch(
                patches.Rectangle(
                    [voc_annotation['bndbox']['xmin'] - 1, voc_annotation['bndbox']['ymin'] - 1],
                    (voc_annotation['bndbox']['xmax'] - voc_annotation['bndbox']['xmin']),
                    (voc_annotation['bndbox']['ymax'] - voc_annotation['bndbox']['ymin']),  # (x,y), width, height
                    fill=False,
                    ec='g',
                )
            )

        else:
            print('repeated box')

#Pulling files
pos_file_names = glob.glob(annotationDir+'*.json')
for filename in pos_file_names:

    # Read annotation.json (Assume format of RectLabel output)
    data = json.load(open(filename))

    xPix, yPix = data.get('image_w_h', [None, None])  # Width and height of original image
    if (xPix != None):  # Then the json file is [for our files] of RectLabel format.

        source_file, cut_id = filename.split('_cut')
        cut_id, ext = cut_id.split('.')

        source_name = source_name.split(annotationDir)[-1]
        source_img = xmlImgDir + source_name + '_cut' + cut_id + '.tif'

        cut_id = int(cut_id)  # convert ID from string

        xTileMax = ((xPix / 2) * xSplits) + (xPix / 2)
        yTileMax = ((yPix / 2) * ySplits) + (yPix / 2)

        # Convert TileID to x,y poss
        x_ind, y_ind = tileID_to_x_y(cut_id, xSplits, ySplits)

        # Find corner pixels of cut piece depending on index
        xMin, xMax, yMin, yMax = get_box_corners(x_ind, y_ind, xPix, yPix)

        #Convert each object from RectLabel to PASCAL_VOC
        for object in data['objects']:
            voc_annotation = RectLabel_2_PASCAL_VOC(object, [xMin, xMax, yMin, yMax], 'crater')
            bbox = checkUniqueBox(voc_annotation, uniqueBoxes)
            if(len(bbox) > 0):
                output['object'].append(voc_annotation)

                ax.add_patch(
                    patches.Rectangle(
                        [voc_annotation['bndbox']['xmin'] - 1, voc_annotation['bndbox']['ymin'] - 1],
                        (voc_annotation['bndbox']['xmax'] - voc_annotation['bndbox']['xmin']),
                        (voc_annotation['bndbox']['ymax'] - voc_annotation['bndbox']['ymin']),  # (x,y), width, height
                        fill=False,
                        ec='g',
                    )
                )

            else:
                print('repeated box')
        #except:
        ##    print(filename+' does not fit format.')

with open(PASCALVOC_filename, 'w') as outfile:
    json.dump(output, outfile)