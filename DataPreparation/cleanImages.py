from PIL import Image
import glob, os
import numpy as np
from osgeo import gdal

#imageDir = '/Users/seabrook/Documents/FDL/FDL-LunarResources/PDS_FILES/LROC_NAC/annotations/'

gdal.UseExceptions()

output_size = [32,32]
stride = np.divide(output_size, 2)
downsample_ratio = 40
imageDir = '/Volumes/DATA DISK/PDS_FILES/LROC_NAC/LRO_New/'
resampledDir = imageDir + 'Resampled/'
#Clean images
pos_file_names = glob.glob(resampledDir + 'Tiles/' + '*.tif')
for filename in pos_file_names:
    ds = gdal.Open(filename)
    image = ds.GetRasterBand(1).ReadAsArray()
    if (image is not None):  # Make sure image load has gone well (some may be unreadable)
        image = np.array(image)  # Convert image into array

        height, width = image.shape  # Read width and height

        originalFilename = imageDir + filename.split('.')[0].split(imageDir+'Tiles/')[1]  # Form original filename from this cut
        originalFilename = originalFilename.split('_d')[0]  # Extract from filename
        originalFilename += '.tif'  # Add file extension
        h1, v1 = filename.split('_x')[1].split('_y')
        v1 = int(v1.split('.')[0])
        h1 = int(h1)
        h2 = h1 + width
        v2 = v1 + height

        ds2 = gdal.Open(originalFilename)  # Open original File
        image2 = ds2.GetRasterBand(1).ReadAsArray()  # Read image data from original file
        if (image2 is not None):
            image2 = np.array(image2)  # If reading original image goes well (it should if the cut did)

            height2, width2 = image2.shape  # read height and width of original image

            # Measure number of cuts made from original image
            # horizontal_cuts
            h_cuts = np.floor_divide(width, stride[1]) - (np.floor_divide(output_size[1], stride[1])) + 1

            # vertical_cuts
            v_cuts = np.floor_divide(height2, stride[0]) - (np.floor_divide(output_size[0], stride[0])) + 1

            h_underflow = width2 - ((h_cuts+2)*stride[1])
            v_underflow = height2 - ((v_cuts+2)*stride[0])
            #The above calculations are valid because of stride being a half cut.

            means = np.mean(image, axis=0)  # Trim black columns
            h_del_ind = np.where(means <= 5)[0]

            shift_hpos = True
            shift_h = 0
            while(shift_hpos):
                if any(h_del_ind == shift_h):
                    #image = np.delete(image, 0, 1) #Don't delete columns in middle of tile
                    shift_h += 1
                else:
                    shift_hpos = False
            h1 += shift_h
            h2 += shift_h

            shift_hrpos = True
            shift_hr = 0
            while (shift_hrpos):
                if any(h_del_ind == (h2-shift_hr-1-h1)):
                    #image = np.delete(image, h2-shift_hr, 1)  # Don't delete columns in middle of tile
                    shift_hr += 1
                else:
                    shift_hrpos = False
            h2 -= shift_hr

            means = np.mean(image, axis=1)  # Trim black rows
            v_del_ind = np.where(means <= 5)[0]

            shift_vpos = True
            shift_v = 0
            while(shift_vpos):
                if any(v_del_ind == shift_v):
                    #image = np.delete(image, 0, 0) #Don't delete rows in middle of tile
                    shift_v += 1
                else:
                    shift_vpos = False
            v1 += shift_v
            v2 += shift_v

            shift_vbpos = True
            shift_vb = 0
            while (shift_vbpos):
                if any(v_del_ind == (v2 - shift_vb-1-v1)):
                    # image = np.delete(image, h2-shift_hr, 1)  # Don't delete columns in middle of tile
                    shift_vb += 1
                else:
                    shift_vbpos = False
            v2 -= shift_vb

            height, width = (v2 - v1), (h2 - h1)  # Read width and height

            #I felt including deletion of columns from main image would cause too much confusion in coordinate system
            #means = np.mean(image2, axis=0)  # Trim black columns
            #del_ind = np.where(means == 0)[0]

            #shift_hpos = True
            #shift_h2 = 0
            #while (shift_hpos):
            #    if any(del_ind == shift_h2):
            #        image = np.delete(image2, 0, 1)  # Don't delete columns in middle of tile
            #        shift_h2 += 1
            #    else:
            #        shift_hpos = False


            # Measure position of cut-in-question
            # horizontal position (divide and round down)
            #   h_pos = np.floor_divide(cut_id, v_cuts)
            # vertical position (remainder after division)
            #   v_pos = np.mod(cut_id, v_cuts)

            # If v_pos is 0, then top is empty
            # If v_pos equals v_cuts, then bottom is empty
            # If h_pos is 0, then left is empty
            # If h_pos equals h_cuts, then right is empty

            if((not((height == 32) & (width == 32)))):
                saveImage = False

                h_diff = 32 - width
                v_diff = 32 - height

                #v1 = v_pos * stride[0]  # top row of cut
                #v2 = v1 + height  # bottom row of cut
                #h1 = h_pos * stride[1]  # left column of cut
                #h2 = h1 + width  # right column of cut

                if h_diff < 0:  # If cut is too wide
                    image = image[:][:h_diff]  # (diff is minus from end)
                    saveImage = True
                if v_diff < 0:  # If cut is too tall
                    image = image[:v_diff][:]  # (diff is minus from end)
                    saveImage = True

                if h_diff > 0:  # If cut is not wide enough
                    if h1 != 0:  # If not leftmost
                        if h2 == width2:  # rightmost cut

                            image = image2[v1:v2,(h1 - h_diff):h2]
                            h1 = h1 - h_diff
                            saveImage = True

                        else:  # middle cut
                            h_add1 = np.floor_divide(h_diff, 2)
                            h_add2 = h_add1 + (h_diff - (2 * h_add1))


                            checkLeft = True
                            h_add = 0
                            while (checkLeft & ((h1 - h_add) > 0)):
                                if ((np.mean(image2[v1:v2, h1 - h_add - 1], axis=0) > 3) and (h_add < h_diff)):
                                    h_add += 1
                                else:
                                    checkLeft = False

                            if(h_add < h_add1):
                                h_add2 += (h_add1 - h_add) #add pixels not found onto search on other side
                            h_add1 = h_add

                            checkRight = True
                            h_add = 0
                            while (checkRight & ((h2 + h_add) < width2)): #Could check all at once rather than stepwise
                                if ((np.mean(image2[v1:v2, h2 + h_add + 1], axis=0) > 3) and (h_add < h_add2)):
                                    h_add += 1
                                else:
                                    checkRight = False

                            if(h_add == h_add2):
                                h_add2 = h_add
                                h_add1 = h_diff - h_add2
                            elif(h_add1 + h_add2 >= h_diff): #h_add2 not reached, but enough pixels found
                                    h_add1 = h_diff - h_add2

                            if(h_add1 + h_add2 == h_diff):
                                image = image2[v1:v2,(h1 - h_add1):h2+h_add2]
                                h1 = h1 - h_add1
                                h2 = h2 + h_add2
                                saveImage = True

                    elif h2 != width2:  # leftmost cut, but not rightmost

                        checkRight = True
                        h_add = 0
                        while (checkRight & ((h2 + h_add) < width2)):
                            if ((np.mean(image2[v1:v2, h2 + h_add + 1], axis=0) > 3) and (h_add < h_diff)):
                                h_add += 1
                            else:
                                checkRight = False

                        if(h_add == h_diff):
                            saveImage = True  # save

                    elif h_underflow >= h_diff: #leftmost and rightmost cut, but buffer remains
                        if (np.mean(np.mean(image2[v1:v2, h1:h2+h_diff], axis=0)) > 3): #If all of buffer is not black
                            image = image2[v1:v2, h1:h2+h_add] #add buffer
                            saveImage = True #save
                            h2 = h2 + h_add

                if v_diff > 0:  # If cut is not tall enough
                    if v1 != 0:  # If not top
                        if v2 == height2:  # bottom cut
                            image = image2[(v1 - v_diff):v2,h1:h2]
                            v1 = v1 - v_diff
                            saveImage = True  # save
                        else:  # middle cut
                            v_add1 = np.floor_divide(v_diff, 2)
                            v_add2 = v_add1 + (v_diff - (2 * v_add1))

                            checkTop = True
                            v_add = 0
                            while (checkTop & ((v1 - v_add) > 0)):
                                if ((np.mean(image2[v1 - v_add, h1:h2]) > 3) and (v_add < v_diff)):
                                    v_add += 1
                                else:
                                    checkTop = False

                            if (v_add < v_add1):
                                v_add2 += (v_add1 - v_add)  # add pixels not found onto search on other side
                            v_add1 = v_add

                            checkBottom = True
                            v_add = 0
                            while (checkBottom & ((v2 + v_add) < height2)):
                                if ((np.mean(image2[v2 + v_add + 1,h1:h2]) > 3) and (v_add < v_add2)):
                                    v_add += 1
                                else:
                                    checkBottom = False

                            if (v_add == v_add2): #if sufficient found
                                v_add2 = v_add
                                v_add1 = v_diff - v_add2
                            elif (v_add1 + v_add2 >= v_diff):  # h_add2 not reached, but enough pixels found overall
                                v_add1 = v_diff - v_add2

                            if (v_add1 + v_add2 == v_diff): #if sum is enough
                                v1 = v1 - v_add1
                                v2 = v2 + v_add2
                                image = image2[v1:v2,h1:h2] #load from image
                                saveImage = True #save

                    elif v2 != height2:  # If top cut, not bottom
                        image = image2[v1:(v2 + v_diff),h1:h2]

                        v2 = v2 + v_diff

                        saveImage = True
                    elif v_underflow >= v_diff: #bottom and top cut, but buffer remains
                        checkBottom = True
                        v_add = 0
                        if(np.mean(np.mean(image2[v1:v2 + v_diff, h1:h2], axis=1)) > 3):
                            image = image2[v1:v2+v_add,h1:h2]
                            saveImage = True

                        v2 = v2 + v_add

                height, width = (v2 - v1), (h2 - h1)  # Read width and height
                if((height == 32) & (width == 32)):
                    if(saveImage == True):
                        output_filename = imageDir + 'Tiles/' + originalFilename.split(imageDir)[1].split('.')[0] + '_x' + str(h1) + '_y' + str(v1) + '.tif'
                        image = Image.fromarray(image)
                        image.save(output_filename)
                else:
                    output_filename = imageDir + 'Tiles/fail' + originalFilename.split(imageDir)[1].split('.')[
                        0] + '_x' + str(h1) + '_y' + str(v1) + '.tif'
                    image = Image.fromarray(image)
                    image.save(output_filename)
                    # os.remove(filename)
            else:
                output_filename = imageDir + 'Tiles/fail' + originalFilename.split(imageDir)[1].split('.')[0] + '_x' + str(h1) + '_y' + str(v1) + '.tif'
                image = Image.fromarray(image)
                image.save(output_filename)