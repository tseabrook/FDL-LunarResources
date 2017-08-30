
import numpy as np
#from PIL import Image
import os
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import math
from osgeo import gdal
#from resampleImage import resample_image
#from scipy.optimize import fsolve, minimize
import glob
import csv



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
            convolution[y,x] = np.sum(target[y:y+kernelSize[0],x:x+kernelSize[1]] * kernel[:,:])

    return convolution
#Convolutions in scipy

#root_dir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/'
root_dir = '/Users/seabrook/Dropbox/fdl_lunar_nac/'


dir_type = ['polar_positive', 'polar_negative']
#dir_type = 'LOLA_DEM'
dir_extension = ['eleni-tim', 'shashi-tim']

for img_dir in dir_type:
    for dir_ext in dir_extension:
        pos_file_names = glob.glob(root_dir + img_dir + '/' + dir_ext + '/' +'*.tif')
        for filename in pos_file_names:

            ds = gdal.Open(filename)
            image = ds.GetRasterBand(1).ReadAsArray()
            if image is None:
                print('broken image:' + filename)
            else:
                #if('NAC' in filename):
                #    image = resample_image(image, 40)  # Reduce scale to match that used by our DNN (0.5m -> 20m resolution, as in DEM)
                #image = np.array(image)

                window_scale = [[8, 8], [15, 15]]  # sliding window sizes
                num_scales = len(window_scale)
                threshold = 0.3

                image_name = filename.split(root_dir + img_dir + '/' + dir_ext + '/')[1].split('.tif')[0]
                image_name, image_id = image_name.split('_d40_')

                #output_name = 'conv_' + image_name
                #output_filename = (root_dir + output_name + '.tif')
                csv_name = 'csv_' + img_dir
                csv_filename = root_dir + csv_name

                fig, axarr = plt.subplots(1, 2)
                axarr[0].imshow(image, cmap='gray')
                image = image.astype(float) - np.min(image.astype(float))
                image = image.astype(float) / (np.max(image.astype(float)) + 1e-8)

                #Grid Search
                #Search over:
                #Foreshortening_angle \in [0,1]
                #Shadow_angle \in [0,1]
                #Foreshortening_degree \in [0,1]
                #Rim_prominence_degree \in [0,1]
                #5*10*5*6 = 1500

                best_score = 0
                variables = [0.375, 0.5, 0.412, 0.5]

                if (os.path.isfile(csv_filename + '.csv')):
                    csv_id = 0
                    while (os.path.isfile(csv_filename + str(csv_id) + '.csv')):
                        csv_id += 1
                    csv_filename += str(csv_id)
                csv_filename += '.csv'
                true_positives, false_positives, true_negatives, false_negatives = 0,0,0,0

                for foreshortening_angle in range(6): #5
                    foreshortening_angle = (foreshortening_angle * 0.1) + 1e-8
                    for shadow_angle in range(11): #10
                        shadow_angle = (shadow_angle * 0.1)
                        for foreshortening_degree in range(13): #5
                            foreshortening_degree = (foreshortening_degree * 0.05) + 0.4
                            for rim_prominence_degree in range(7): #6
                                rim_prominence_degree = (rim_prominence_degree * 0.2) + 0.2
                                for scale in window_scale:
                                    kernel = buildCraterConvolutionKernel(scale[0], scale[1], foreshortening_angle, shadow_angle, foreshortening_degree, rim_prominence_degree)
                                    #axarr[1].imshow(kernel, cmap='gray')
                                    convolution = slidingConvolution(image, kernel)
                                    score = np.max(convolution)
                                    if score > best_score:
                                        variables = [foreshortening_angle, shadow_angle, foreshortening_degree, rim_prominence_degree]
                                        best_score = score
                                    if best_score > threshold:
                                        break
                                if best_score > threshold:
                                    break
                            if best_score > threshold:
                                break
                        if best_score > threshold:
                            break
                    if best_score > threshold:
                        break

                if best_score > threshold:
                    with open(csv_filename, 'a') as csvfile:
                        craterwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                        craterwriter.writerow([img_dir, dir_ext, image_id, '1'])
                        if(dir_ext == 'Craters'):
                            true_positives += 1
                        else:
                            false_positives += 1
                else:
                    with open(csv_filename, 'a') as csvfile:
                        craterwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                        craterwriter.writerow([img_dir, dir_ext, image_id, '0'])
                        if (dir_ext == 'Not Crater'):
                            true_negatives += 1
                        else:
                            false_negatives += 1
                    #crater has been detected

        not_crater_accuracy = float(true_negatives) / float(true_negatives + false_negatives)
        crater_accuracy = float(true_positives) / float(true_positives + false_positives)
        overall_accuracy = float(true_negatives + true_positives) / float(true_negatives + false_negatives + true_positives + false_positives)

    with open(csv_filename, 'a') as csvfile:
        craterwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        craterwriter.writerow(['true_positives', true_positives])
        craterwriter.writerow(['false_positives', false_positives])
        craterwriter.writerow(['true_negatives', true_negatives])
        craterwriter.writerow(['false_negatives', false_negatives])
        craterwriter.writerow(['not_crater_accuracy', not_crater_accuracy])
        craterwriter.writerow(['crater_accuracy', crater_accuracy])
        craterwriter.writerow(['overall_accuracy', overall_accuracy])
        print('Results for ' + dir_ext)
        print('++: ' + str(true_positives) + ' -+: ' + str(false_positives) + ' +-: ' + str(true_negatives) + ' --: ' + str(false_negatives))
        print('Crater accuracy: ' + str(crater_accuracy))
        print('Not crater accuracy: ' + str(not_crater_accuracy))
        print('Overall accuracy: ' + str(overall_accuracy))