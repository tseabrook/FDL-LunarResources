import glob, os
from PIL import Image
import numpy as np
from lxml.etree import fromstring, tostring
import matplotlib.pyplot as plt
from skimage import io
from osgeo import gdal
import json
from pprint import pprint
import xml.etree.ElementTree as ET


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
# The LOLA South-Pole NAC

def tileID_to_x_y(id, width, height, order = 'TopBottom'):
    # Find indexes of cut piece depending on tiling order
    if (order == 'TopBottom'):
        x = np.floor_divide(id, height)
        y = np.mod(cut_id, height)
    elif (order == 'LeftRight'):
        y = np.floor_divide(id, width)
        x = np.mod(cut_id, width)
    else:
        print('expected order: TopBottom or LeftRight, assumed former.')
        x = np.floor_divide(id, height)
        y = np.mod(cut_id, height)

    return x, y

def get_box_corners(x,y,width,height):
    xMin = x*(width/2)
    xMax = xMin + width
    yMin = y*(height/2)
    yMax = yMin + height
    return xMin, xMax, yMin, yMax

def RectLabel_2_PASCAL_VOC(object, corners, name, difficult = False):
    x, y, w, h = object['x_y_w_h']
    xMin, xMax, yMin, yMax = corners
    voc_annotation = {
        "bndbox": {
            "xmax": xMax + x + w,
            "xmin": xMin + x,
            "ymax": yMax + y + h,
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

annotationDir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_NAC/annotations/'
uniqueBoxes = []
num_craters = 0

pos_file_names = glob.glob(annotationDir+'*.xml')
for filename in pos_file_names:

    tree = ET.parse(filename)
    root = tree.getroot()
    source_name = root.find('filename').text.split('.')[0]

    output_filename = annotationDir+source_name+'_VOC.json'

    if (os.path.isfile(output_filename)):
        output = json.load(open(output_filename))
        uniqueBoxes = output['object']
        num_craters = num_craters + len(uniqueBoxes)
    else:
        output = {'object': []}
        uniqueBoxes = []

    for object in root.findall('object'):
        bndbox = object._children[4]
        voc_annotation = {
            "bndbox": {
                "xmax": bndbox._children[2].text,
                "xmin": bndbox._children[0].text,
                "ymax": bndbox._children[3].text,
                "ymin": bndbox._children[1].text,},
            "difficult": object._children[3].text,
            "name": object._children[0].text,
        }
        bbox = checkUniqueBox(voc_annotation, uniqueBoxes)
        if (len(bbox) > 0):
            output['object'].append(voc_annotation)
        else:
            print('repeated box')
        num_craters += 1

    with open(output_filename, 'w') as outfile:
        json.dump(output, outfile)

#Pulling files
pos_file_names = glob.glob(annotationDir+'*[0-9]*.json')
for filename in pos_file_names:
    if not filename.endswith('_VOC.json'):

        data = json.load(open(filename))

        xPix, yPix =  data.get('image_w_h', [None, None])  # Width and height of original image
        if(xPix != None): #Then the json file is [for our files] of RectLabel format.

            output_filename = filename.split('.')[0]+'_VOC.json'

            if (os.path.isfile(output_filename)):
                output = json.load(open(output_filename))
                uniqueBoxes = output['object']
                num_craters = num_craters + len(uniqueBoxes)
            else:
                output = {'object': []}
                uniqueBoxes = []

            #Find corner pixels of cut piece depending on index
            xMin, xMax, yMin, yMax = get_box_corners(0,0,xPix,yPix)

            #Convert each object from RectLabel to PASCAL_VOC
            for object in data['objects']:
                voc_annotation = RectLabel_2_PASCAL_VOC(object, [xMin, xMax, yMin, yMax], 'crater')
                bbox = checkUniqueBox(voc_annotation, uniqueBoxes)
                if(len(bbox) > 0):
                    output['object'].append(voc_annotation)
                else:
                    print('repeated box')
                num_craters += 1

            with open(output_filename, 'w') as outfile:
                json.dump(output, outfile)

    print num_craters