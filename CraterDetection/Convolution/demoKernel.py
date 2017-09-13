#Casey Handmer's Adaptive Crater Convolution Kernel
#caseyhandmer@gmail.com
#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import math
from osgeo import gdal
#from resampleImage import resample_image
#from scipy.optimize import fsolve, minimize
import glob
import csv

#demoKernel generates images of the kernel adaption to be used in presentation

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
    kernel = kernel[:] - kernel.min()
    kernel = kernel[:]/(kernel.max()+ 1e-8)
    return kernel

def slidingConvolution(target, kernel, step=1):
    targetSize = target.shape
    kernelSize = kernel.shape
    convolution = np.ones((targetSize[0]-kernelSize[0]+1, targetSize[1]-kernelSize[1]+1))
    for y in range(0, targetSize[0]-kernelSize[0] + 1, step):
        for x in range(0, targetSize[1] - kernelSize[1] + 1, step):
            convolution[y,x] = np.mean(target[y:y+kernelSize[0],x:x+kernelSize[1]] * kernel[:,:])

    return convolution



thisDir = os.path.dirname(os.path.abspath(__file__))
demoDir = os.path.join(thisDir, 'Kernel')

if(not os.path.isdir(demoDir)): #SUBJECT TO RACE CONDITION
    os.makedirs(demoDir)

window_scale = [[15, 15]]  # sliding window sizes
num_scales = len(window_scale)

id = 0

for scale in window_scale:
    for rim_prominence_degree in range(5):  # 6
        rim_prominence_degree = (rim_prominence_degree * 0.2) + 0.2
        for foreshortening_angle in range(6):  # 5
            foreshortening_angle = (foreshortening_angle * 0.1) + 1e-8
            for shadow_angle in range(11):  # 10
                shadow_angle = (shadow_angle * 0.1) + 1e-8
                for foreshortening_degree in range(13): #5
                    foreshortening_degree = (foreshortening_degree * 0.05) + 0.4
                    kernel = buildCraterConvolutionKernel(scale[0], scale[1], foreshortening_angle, shadow_angle,
                                                  foreshortening_degree,
                                                  rim_prominence_degree)
                    image = Image.fromarray(((kernel*255)//1).astype(np.uint8))
                    image.save(demoDir + 'kernel_'+str(id).zfill(5)+'.png')
                    id += 1