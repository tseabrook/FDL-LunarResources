from PIL import Image
import glob, os
import numpy as np
from osgeo import gdal

#Check images takes tiles that have been previously cut
#And asserts whether they are the correct size
#If the size of the tile is not correct, then the source image is loaded
#So that adjacent pixels can be added to the tiles.
#This was necessary due to a coding error in the img_split script.

#CHECK IMAGES is written to operate on cuts numbered incrementally
#CLEAN IMAGES is written to operate on cuts identifed with x1 and y1 coordinates

output_size = [32,32]
stride = np.divide(output_size, 2)

imageDir = '/Volumes/DATA DISK/PDS_FILES/LROC_NAC/Repo/imgs/LROC_NAC/Resampled/40/'
sourceDir = '/Users/seabrook/Dropbox/fdl_lunar_nac/polar_positive/eleni/misnamed/'
targetDir = '/Users/seabrook/Dropbox/fdl_lunar_nac/polar_positive/eleni/renamed/'
#cratersDir = imageDir + 'Tiles/Craters/'
#cratersDir = '/Volumes/DATA DISK/PDS_FILES/Elenizdabest/Unlabelled 3/Big crater'
#notCratersDir = imageDir + 'Tiles/Not Crater/'
#notCratersDir = imageDir + '/Volumes/DATA DISK/PDS_FILES/Elenizdabest/Unlabelled 3/Not Cratered'
#notSureDir = imageDir + 'Tiles/Not Sure/'

#misshapenDir = imageDir + 'Tiles/Misshapen/'

pos_file_names = glob.glob(sourceDir + '*.tif')
for filename in pos_file_names:
    ds = gdal.Open(filename)
    image = ds.GetRasterBand(1).ReadAsArray()
    if (image is not None):  # Make sure image load has gone well (some may be unreadable)
        image = np.array(image)  # Convert image into array

        height, width = image.shape  # Read width and height

        originalFilename = imageDir + filename.split('.')[0].split(sourceDir)[1]  # Form original filename from this cut
        originalFilename, cut_id = originalFilename.split('_cut')  # Extract cut_id from filename
        cut_id = int(cut_id)  # Cast cut_id from string to int
        originalFilename += '.tif'  # Add file extension
        ds2 = gdal.Open(originalFilename)  # Open original File
        image2 = ds2.GetRasterBand(1).ReadAsArray()  # Read image data from original file
        if (image2 is not None):
            image2 = np.array(image2)  # If reading original image goes well (it should if the cut did)

            originalFilename = originalFilename.split('_d')[0]
            originalFilename += '.tif'

            height2, width2 = image2.shape  # read height and width of original image

            # Measure number of cuts made from original image
            # horizontal_cuts
            h_cuts = np.multiply(np.floor_divide(width2, output_size[1]),
                                 np.divide(output_size[1], stride[1]))
            # Above is incorrect but is what was used, in future use below
            #h_cuts = np.floor_divide(width, stride[1]) - (np.floor_divide(output_size[1], stride[1])) + 1

            # vertical_cuts
            v_cuts = np.multiply(np.floor_divide(height2, output_size[0]),
                                 np.divide(output_size[0], stride[0]))
            #Above is incorrect but is what was used, in future use below
            #v_cuts = np.floor_divide(height2, stride[0]) - (np.floor_divide(output_size[0], stride[0])) + 1

            h_underflow = width2 - ((h_cuts+2)*stride[1])
            v_underflow = height2 - ((v_cuts+2)*stride[0])
            #The above calculations are valid because of stride being a half cut.

            #I messed up the cut_id calculation for these tiles.
            #((h_pos * h_cuts) + v_pos) was used, rather than ((h_pos * v_cuts) + v_pos)
            #For this reason, we must search for the tile, as it became a many to one.
            #Also, variable shadows mean sometimes the rightmost tile may not have overwritten the tile to the top-left
            #So one cannot simply walk into mordor... or take the righthand tiles for indexes of cut_id > h_cuts

            searchImg = True
            h_pos = 0
            while((searchImg) & (h_pos < (h_cuts+1))):
                v_pos = cut_id - (h_cuts * h_pos)
                v1 = v_pos * stride[0]  # top row of cut
                v2 = v1 + height  # bottom row of cut
                h1 = h_pos * stride[1]  # left column of cut
                h2 = h1 + width  # right column of cut
                if((v2 <= height2) & (h2 <= width2)):
                    if(np.mean((image == image2[v1:v2, h1:h2]).astype(np.uint8)) > 0.8):
                        searchImg = False
                    else:
                        h_pos += 1
                else:
                    h_pos += 1

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


            if(not((height == 32) & (width == 32))):
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
                    if h_pos != 0:  # If not leftmost
                        if h_pos == h_cuts:  # rightmost cut
                            checkRight = True
                            h_add = 0
                            while((checkRight) & (h_add < h_underflow)):
                                if((np.mean(image2[v1:v2,h2+h_add+1], axis=0) > 3) and (h_add < h_diff)):
                                    h_add += 1
                                else:
                                    checkRight = False
                            image = image2[v1:v2,(h1 - (h_diff - h_add)):h2 + h_add]
                            h1 = h1 - (h_diff - h_add)
                            h2 = h2 + h_add
                            saveImage = True
                        else:  # middle cut
                            h_add1 = np.floor_divide(h_diff, 2)
                            h_add2 = h_add1 + (h_diff - (2 * h_add1))

                            checkRight = True
                            h_add = 0
                            while (checkRight & ((h2 + h_add) < width2)):
                                if ((np.mean(image2[v1:v2, h2 + h_add + 1], axis=0) > 3) and (h_add < h_add2)):
                                    h_add += 1
                                else:
                                    checkRight = False

                            h_add1 += (h_add2 - h_add)
                            h_add2 = h_add

                            # No need to check if still in bounds, since h_diff < output_size
                            image = image2[v1:v2,(h1 - h_add1):h2+h_add2]

                            h1 = h1 - h_add1
                            h2 = h2 + h_add2

                            saveImage = True
                    elif h_pos != h_cuts:  # leftmost cut, but not rightmost
                        image = image2[v1:v2,h1:(h2+h_diff)]

                        h2 = h2 + h_diff
                        saveImage = True  # save
                    elif h_underflow >= h_diff: #leftmost and rightmost cut, but buffer remains
                        if (np.mean(np.mean(image2[v1:v2, h1:h2+h_diff], axis=0)) > 3): #If all of buffer is not black
                            image = image2[v1:v2, h1:h2+h_add] #add buffer
                            saveImage = True #save
                            h2 = h2 + h_add

                if v_diff > 0:  # If cut is not tall enough
                    if v_pos != 0:  # If not top
                        if v_pos == v_cuts:  # bottom cut

                            checkBottom = True
                            v_add = 0
                            while (checkBottom & (v_add < v_underflow)):
                                if ((np.mean(image2[v2 + v_add + 1,h1:h2]) > 3) and (v_add < v_diff)):
                                    v_add += 1
                                else:
                                    checkBottom = False
                            image = image2[(v1 - v_diff + v_add):v2+v_add,h1:h2]
                            v1 = v1 - (v_diff - v_add)
                            v2 = v2 + v_add
                            saveImage = True  # save
                        else:  # middle cut
                            v_add1 = np.floor_divide(v_diff, 2)
                            v_add2 = v_add1 + (v_diff - (2 * v_add1))

                            checkBottom = True
                            v_add = 0
                            while (checkBottom & ((v2 + v_add) < height2)):
                                if ((np.mean(image2[v2 + v_add + 1,h1:h2]) > 3) and (v_add < v_add2)):
                                    v_add += 1
                                else:
                                    checkBottom = False

                            v_add1 += (v_add2 - v_add)
                            v_add2 = v_add

                            # No need to check if still in bounds, since h_diff < output_size
                            image = image2[(v1 - v_add1):v2 + v_add2,h1:h2]

                            v1 = v1 - v_add1
                            v2 = v2 + v_add

                            saveImage = True #save
                    elif v_pos != v_cuts:  # If top cut, not bottom
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
                        output_filename = targetDir + originalFilename.split(imageDir)[1].split('.')[0] + '_d40' + '_x' + str(h1) + '_y' + str(v1) + '.tif'
                        image = Image.fromarray(image)
                        image.save(output_filename)
                else:
                    output_filename = targetDir + 'failReshape' + originalFilename.split(imageDir)[1].split('.')[
                        0] + '_d40' + '_x' + str(h1) + '_y' + str(v1) + '.tif'
                    image = Image.fromarray(image)
                    image.save(output_filename)
                    # os.remove(filename)
            else:
                output_filename = targetDir + originalFilename.split(imageDir)[1].split('.')[0] + '_d40' + '_x' + str(h1) + '_y' + str(v1) + '.tif'
                image = Image.fromarray(image)
                image.save(output_filename)
        else:
            output_filename = targetDir + 'failFind' + originalFilename.split(imageDir)[1].split('.')[0] + '_d40' + '_x' + str(
                h1) + '_y' + str(v1) + '.tif'
            image = Image.fromarray(image)
            image.save(output_filename)