#Casey Handmer's Adaptive Crater Convolution Kernel
#Written in Mathematica
#caseyhandmer@gmail.com
#Translated by Timothy Seabrook with additions
#timothy.seabrook@cs.ox.ac.uk

#This script in its current form doesn't work very well...
#Do try and improve it!

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from osgeo import gdal
from scipy.optimize import fsolve, minimize
import glob
import os, sys

sys.path.append('../../DataPreparation/LROC_NAC/')

#from NAC2_Resample import resample_image

#Define the image detection kernels.

#Crater1[scale_] := Table[Cos[ArcTan[x,y]] (Sqrt[x^2 + y^2] - 3.166) Exp[-(Sqrt[x^2 + y^2] - 3) ^2],
#   {x, -5.001, 5, 1. /scale}, {y, -5., 5, 1. /scale}];
#Crater2[scale_] := Table[Sin[ArcTan[x,y]] (Sqrt[x^2 + y^2] - 3.166) Exp[-(Sqrt[x^2 + y^2] - 3) ^2],
#   {x,-5.001, 5,1./scale}, {y, -5.,5,1./scale}];

#def craterConvolution(x,y,scale):
#    a1 = np.cos(np.atan(x,y))
#    a2 = np.sqrt(np.power(x,2)+np.power(y,2)) - 3.166
#    a3 = np.exp(-np.power((np.sqrt(np.power(x,2)+np.power(y,2))-3),2))
#    b1 = np.sin(np.atan(x,y))
#    b2 = a2
#    b3 = a3
#    a = np.multiply(np.multiply(a1,a2), a3)
#    b = np.multiply(np.multiply(b1,b2), b3)
#    return [a, b]


#They are adapted from a standard Sobel filter designed to detect adjacent edges at a particular scale.
# We have two, one for each direction. Their scale is freely determined.

#{MatrixPlot{Crater1[10]], MatrixPlot[Crater2[10]]}
#

#def plotCrater(crater):
#    fig, ax = plt.subplots(1)
#    ax.plot(crater)

#This function combines the convolution of the scaled crater kernels with the main image,
# according to a particular angle corresponding to the direction of local sunlight.
# This is set by hand, though automated detection isnt impossible.
#
#
#FindCratersAngle[pic_, scale_, angle_] =
#   ImagePad[Image[#/Max[#]], (Length[ImageData[pic]] - Length[#])/2 &[
#       ListConvolve[Crater1[scale], ImageData[pic]] Cos[angle] +
#           ListConvolve[Crater2[scale], ImageData[pic]] Sin[angle]]
#

#def findCratersAngle(image, scale, angle, conv1, conv2):
#    iter = np.floor_divide(image.shape, scale)
#    for i in range(iter[0]):
#        for j in range(iter[1]):
#           output = np.multiply(image[np.multiply(j,scale):np.multiply(j+1,scale),
#            np.multiply(i, scale):np.multiply(i+1, scale)], conv1(angle) #conv1(angle) is conv1 rotated by angle
#            ) + np.multiply(image[np.multiply(j, scale):np.multiply(j + 1, scale),
#            np.multiply(i, scale):np.multiply(i + 1, scale)], conv2(angle)) #conv2(angle) is conv1 rotated by angle
#    return output

#This takes all the craters discovered at different scales and layers the data.
#For size/location data, different processing of the results are needed with no substantial increase in difficulty.
#Theres also some thresholding happening in the Tanh term to make the image less difficult to see.
#
#Image[Transpose[{ImageData[ImageAdd[
#   Table[Binarize[FindCratersAngle[picbw, i, pi/2], 0.7+0.2 Tanh[(i-10)/3]],
#       {i,1,25}]]]], 0 ImageData[picbw], ImageData[picbw]}, {3,1,2}]]
#

#def layerCraters(image, craters):
#    for crater in craters:
        #bbox = bound(crater)
        #image = image + bbox

#
#These functions locate the centers of craters at each test scale,
#and provide a primitive to plot coloured boxes around these locations of the appropriate size.
#
        #CraterMorphGroups[im_,scale_] := MorphologicalComponents[
#   Binarize[FindCratersAngle[im,#,pi/2], 0.7+0.2 Tanh[(#-10)/3]]] & [scale]
#CraterCenters[im_,scale_i] := Table[Mean[N[PixelValuePositions[Image[#],i]]],
#   {i,1,Max[#]}} &[CraterMorphGroups[im,scale]]
#CS[centers_,scale_] := Table[Line[Table[centers[[i]] +6scale{Cos[th],Sin[th]},
#   {th, pi/4.,2.3pi, pi/2}]], {i,1,Length[centers]}]

#def plotCraterBbox(axes, picbw):
#    axes.add_patch(
#        patches.Rectangle(
#            [xmin, ymin],
#            (width),
#            (height),  # (x,y), width, height
#            fill=False
#        )
#    )

#Combine the whole lot to see the results.
# This sort of function can be used to generate training sets to find craters with more sophisticated CNNs.
#
#craterlocs=Graphics[John[{Thick},
#   Flatten[Table[{Hue[0.05scale], CS[CraterCenters[picbw,scale],scale]},
#       {scale, Table[2.^i,{i,0,4.5,1/3}]}]]]]

#-----------------------------------------------------------------------#
# From post-it note written by Casey Handmer
#
# rho = sqrt[(x cos 2 pi alpha + y sin 2 pi alpha )^2
#       + gamma^-2 (-x sin 2pi alpha + y cos 2 pi alpha)^2]
# K = cos(tan^-1(y/x) + 2pi beta) * (rho - 4 + delta)x * e^-(rho - 3)^2
#
#
# K defined for x, y \in [-5, 5] where [-5, 5] is a sample density which determines the kernel scale
# alpha, beta, gamma, delta \in [0,1]
# alpha: foreshortening angle
# beta: shadow angle
# gamma: foreshortening degree
# delta: rim prominence parameter
#
def buildCraterConvolutionKernel(scaleX, scaleY, foreshortening_angle, shadow_angle, foreshortening_degree, rim_prominence_degree):

    alpha, beta, gamma, delta = float(foreshortening_angle), float(shadow_angle), float(foreshortening_degree), float(rim_prominence_degree)
    #Alpha 0.375 is approx 1:1
    #Beta in [0,1] * 2PiRadian
    #Gamma in [0,1]
    #Delta in [0,1]


    kernel = np.zeros((2*scaleX+1,2*scaleY+1))
    sample_density = [5./scaleX, 5./scaleY]

    for thisX in range(-scaleX, scaleX+1):
        for thisY in range(-scaleY, scaleY+1):
            x = (sample_density[0]*thisX)
            y = (sample_density[1]*thisY)
            rho =   ((((x*math.cos(2*math.pi*alpha) + y*math.sin(2*math.pi*alpha))**2) + \
                    (gamma**-2) * ((-x * math.sin(2*math.pi*alpha) + y * math.cos(2*math.pi*alpha))**2))+1e-8)**(0.5)
            kernel[thisX+scaleX,thisY+scaleY] = (math.cos(math.atan(y/(x+1e-8)) + 2*math.pi*beta)) * (rho - 4 + delta)*x *math.exp(-(rho-3)**2)
    return kernel

def slidingConvolution(target, kernel, step=1):
    targetSize = target.shape
    kernelSize = kernel.shape
    convolution = np.ones((targetSize[0]-kernelSize[0]+1, targetSize[1]-kernelSize[1]+1))
    for y in range(0, targetSize[0]-kernelSize[0] + 1, step):
        for x in range(0, targetSize[1] - kernelSize[1] + 1, step):
            convolution[y,x] = np.mean(target[y:y+kernelSize[0],x:x+kernelSize[1]] * kernel[:,:])

    return convolution

def plotCraterBbox(axes, bbox):
    axes.add_patch(
        patches.Rectangle(
            [bbox[0], bbox[1]],
            (bbox[2]),
            (bbox[3]),  # (x,y), width, height
            fill=False,
            ec='g',
        )
    )


def connectedComponents(binary_image):
    # Algorithm 1: Connected Components
    # Initialize all cells in Map to unlabeled2:
    #    while num(unlabeled) > 0
    #        cseed = next_unlabeled
    #        Setlabel(cseed) <- uniqueLabel
    #        C <- FLOODFILL3D(cseed)
    #        Setlabel(c) forall c in C
    #    end
    # function FLOODFILL3D(seed)
    #    return the set C of all cells connected to seed
    # end
    h, w = binary_image.shape[0], binary_image.shape[1] #Size of search area

    labels = np.zeros((h,w), dtype=np.uint32) #uint32 covers 0 to 4,294,967,295

    candidates = np.where(binary_image) #find all pixels that satisfy illumination and DTE
    valid_pixels = np.zeros((h,w), dtype=bool)
    valid_pixels[candidates[0],candidates[1]] = True

    #  #Become candidates for connected components search
    nextLabel = 0 #Region ID (0 means unlabelled)
    checkList = [] #Initialise checklist, contains pixels for neighbourhood checks
    num_total = candidates[0].size #Count number of valid unlabelled pixels
    num_complete = 0 #Initialise counter
    next_ind = 0

    #BEGIN CONNECTED COMPONENTS ALGORITHM

    while(num_complete < num_total):
        nextLabel += 1 #Increment label class ID
        y, x = candidates[0][next_ind], candidates[1][next_ind]
        while(labels[y,x] != 0):
            next_ind += 1
            y, x = candidates[0][next_ind], candidates[1][next_ind]

        labels[y,x] = nextLabel #Add next pixel to the new label class

        if checkList.__len__() == 0: #Create a list of pixels for FloodFill neighbour checking
            checkList = [[y, x]]
        else:
            checkList = checkList.append([y, x])

        #BEGIN FLOODFILL ALGORITHM
        while checkList.__len__() > 0: #Whilst there are qualifying pixels in this iteration of FloodFill
            y, x = checkList.pop() #Take pixel from checklist, to find qualifying neighbours
            num_complete += 1 #update count for timer

            #BEGIN LOCATION SPECIFIC NEIGHBOUR INDEXING
            if x > 0:
                if x < (w - 1): #middle column
                    xmin, xmax = 0, 3
                else: #rightmost column
                    xmin, xmax = 0, 2
            else: #leftmost column
                xmin, xmax = 1, 3

            if y > 0:
                if y < (h - 1): #middle row
                    ymin, ymax = 0, 3
                else: #bottom row
                    ymin, ymax = 0, 2
            else: #top row
                ymin, ymax = 1, 3
            #END LOCATION SPECIFIC NEIGHBOUR INDEXING

            #BEGIN NEIGHBOUR TRAVERSABILITY CHECK
            for i in range(xmin, xmax):
                for j in range(ymin, ymax): #for all neighbouring pixels
                    if (not((j == 1) & (i == 1))): #not including current pixel
                        if valid_pixels[y+j-1,x+i-1] == True: #and only considering unlabeled pixels
                            #if(map.illuminationMap[y+j-1,x+i-1] == 1) & (map.DTEMap[y+j-1,x+i-1] == 1): #not necessary, already checked in precompute
                            labels[y+j-1,x+i-1] = nextLabel
                            valid_pixels[y+j-1,x+i-1] = False
                            checkList.append([y+j-1,x+i-1])
            #END NEIGHBOUR TRAVERSABILITY CHECK
        #END FLOODFILL ALGORITHM
        #seeds = np.where(labels == 0) #Reset candidate seeds
    #END CONNECTED COMPONENTS ALGORITHM
    return labels #return labels and count


def main():
    #MAIN Function uses optimisation function, which doesn't perform as well/as reliably as grid search
    #Examine the 'testConvolutionMethod' script for a grid-search approach.
    thisDir = os.path.dirname(os.path.abspath(__file__))
    rootDir = os.path.join(thisDir, os.pardir, os.pardir)
    dataDir = os.path.join(rootDir, 'Data')
    NACDir = os.path.join(dataDir, 'LROC_NAC', 'South_Pole', 'Resampled')

    #root_dir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_NAC/annotations/'

    pos_file_names = glob.glob(NACDir+'*.tif')
    for filename in pos_file_names:

        ds = gdal.Open(filename)
        image = ds.GetRasterBand(1).ReadAsArray()
        if image is None:
            print('broken image:' + filename)
        else:
            #image = resample_image(image, 40) #Reduce scale to match that used by our DNN (0.5m -> 20m resolution, as in DEM)
            image = np.array(image)

            window_scale = [[2,2],[4,4],[8,8],[16,16]] #sliding window sizes
            num_scales = len(window_scale)
            threshold = 0.4

            image_name = filename.split(NACDir)[1].split('.tif')[0]
            output_name = 'conv_' + image_name
            output_filename = os.path.join(NACDir, output_name + '.tif')


            fig,axarr = plt.subplots(1,2)
            axarr[0].imshow(image, cmap='gray')
            image = image.astype(float) - np.min(image.astype(float))
            image = image.astype(float) / np.max(image.astype(float))

            def objective(x, image, scale):
                kernel = buildCraterConvolutionKernel(scale[0],scale[1],x[0],0.875,x[1],0.5)
                convolution = slidingConvolution(image, kernel)
                score = -np.sum(convolution)
                return score

            #xopt = fmin_bfgs(rosen, x0, fprime=rosen_der)

            bnds = ((0.3, 0.44), (0. + 1e-8, 1.))

            # OPTIMISING OVER: foreshortening_angle, shadow_angle
            res = minimize(objective, (0.375,1.), args=(image, [window_scale[0][0],window_scale[0][1]]), bounds=bnds,
                           method='L-BFGS-B', options={'disp': True})

            #fsolve(func, 0.3)
            craters = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

            for scale in window_scale:
                kernel = buildCraterConvolutionKernel(scale[0],scale[1],res.x[0],0.5,res.x[1],0.5)
                axarr[1].imshow(kernel, cmap='gray')

                convolution = slidingConvolution(image, kernel)
                #Need to connect components
                score = np.sum(convolution)
                craters[0:-kernel.shape[0]+1,0:-kernel.shape[1]+1] += (convolution > threshold).astype(bool)

            craters = connectedComponents(craters)

            image = ((image * 255)//1).astype(np.uint8)

            rgb_image = np.dstack((np.dstack((image, image)), image))  # 3 channel

            num_craters = np.max(craters)
            crater_box = [None]*num_craters
            for i in range(1, num_craters+1):
                indexes = [np.where(craters == i)]
                crater_box[i-1] = [indexes[0][0].min(), indexes[0][0].max(), indexes[0][1].min(), indexes[0][1].max()]
                bbox = [indexes[0][1].min(), indexes[0][0].min(), indexes[0][1].max()-indexes[0][1].min(), indexes[0][0].max()-indexes[0][0].min()]
                plotCraterBbox(axarr[0], bbox)

                for y in range(crater_box[i-1][0] - 1, crater_box[i-1][1] + 1):
                    rgb_image[y, crater_box[i-1][2] - 1: crater_box[i-1][3]] = [255, 0, 0]

            image = Image.fromarray(rgb_image)
            if (image != None):
                image.save(output_filename)

    print('done')

if __name__ == '__main__':
    main()
