import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from scipy.sparse import csr_matrix

from split_and_merge import split_and_merge

import glymur



def edgeCluster(edges, max_step):
    #edgeCluster algorithm
    #Perform a walk from each edge pixel
    #max_step determines how far a pixel can be for it
    # to be considered part of the same edge

    w, h = edges.shape[1], edges.shape[0] #size of search area

    labels = np.zeros((h, w), dtype=np.uint32) #uint32 covers 0 to 4,294,967,295
    data = np.where(edges)

    nextLabel = 0 #Region ID (0 means unlabelled)
    checkList = [] #Initialise checklist, contains pixels for neighbourhood traversability checks

    num_total = len(data[0]) #Count number of valid unlabelled pixels
    num_complete = 0 #Initialise counter
    ind = 0

    #BEGIN CONNECTED COMPONENTS ALGORITHM

    while(num_complete < num_total):
        nextLabel += 1 #Increment label class ID
        y, x = data[0][ind], data[1][ind]
        while(labels[y,x] != 0):
            ind += 1
            y, x = data[0][ind], data[1][ind]

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
            if x > (max_step-1):
                xmin = -max_step
                if x < (w - max_step): #middle column
                    xmax = 1+max_step
                else: #rightmost column
                    xmax = 1+(w-x-1)
            else: #leftmost column
                xmax = 1+max_step
                xmin = -x

            if y > (max_step-1):
                ymin = -max_step
                if y < (h - max_step): #middle row
                    ymax = 1+max_step
                else: #bottom row
                    ymax = 1+(h-y-1)
            else: #top row
                ymax = 1+max_step
                ymin = -y
            #END LOCATION SPECIFIC NEIGHBOUR INDEXING

            #BEGIN NEIGHBOUR TRAVERSABILITY CHECK
            for i in range(xmin, xmax):
                for j in range(ymin, ymax): #for all neighbouring pixels
                    if (((j == 0) & (i == 0))!=True): #not including current pixel
                        if(labels[y + j, x + i] == 0):
                            if edges[y+j,x+i] == True: #and only considering unlabeled pixels
                                labels[y+j,x+i] = nextLabel
                                checkList.append([y+j,x+i])
            #END NEIGHBOUR TRAVERSABILITY CHECK
        #END FLOODFILL ALGORITHM
        #seeds = np.where(labels == 0) #Reset candidate seeds
    #END CONNECTED COMPONENTS ALGORITHM

    cols = np.arange(labels.size)
    M = csr_matrix((cols, (labels.ravel(), cols)),
                      shape=(labels.max() + 1, labels.size))
    indices = [np.unravel_index(row.data, labels.shape) for row in M]
    counts = np.zeros((np.max(labels)+1))
    for i in range(np.max(labels)+1):
        counts[i] = indices[i][0].size
    return indices, counts

    #return labels #return labels and count


filename = 'imgs/testdata/m108898482_cdr_jp2_p'

for i in range(64):
    curr_filename = filename+str(i+1)+'.jp2'


    # Load picture and detect edges
    image = glymur.Jp2k(curr_filename)[:]
    # Low threshold and High threshold represent number of pixels that may be skipped to make a line [4, 60 seems good]
    # Sigma represents the width of the guassian smoothing kernel [3 seems good]
    edges = canny(image, sigma=3, low_threshold=4, high_threshold=60)
    #fig, axarr = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    #axarr[1].imshow(image, cmap=plt.cm.gray)
    #plt.show()

    lines, counts = edgeCluster(edges,3)
    #segments = np.zeros(len(lines))
    segmentParent = np.zeros(len(lines), dtype=int)
    #data = np.where(edges)
    for i in range(1,len(lines)):
        if i == 1:
            segments = split_and_merge(lines[i], 1)
            segmentParent[i] = len(segments)
        else:
            segments = np.hstack((segments, split_and_merge(lines[i], 1)))
            segmentParent[i] = segments.size

    cm = plt.get_cmap('gist_rainbow')
    fig, axarr = plt.subplots(ncols=3, nrows=1)
    axarr[0].imshow(edges, cmap=plt.cm.gray)
    axarr[1].imshow(image, cmap=plt.cm.gray)
    axarr[1].set_color_cycle([cm(1. * i / 20) for i in range(20)])
    for data in lines:
        y, x = data
        axarr[1].scatter(x, y, alpha=0.8, edgecolors='none', s=1)


    axarr[2].imshow(image, cmap=plt.cm.gray)
    #For every grouped line
    for i in range(1,len(lines)):
        first = segmentParent[i-1]
        last = segmentParent[i]

        #For every segment of line
        for j in range(first,last):
            minX = np.min(segments[j].data[0])
            maxX = np.max(segments[j].data[0])
            minY = np.min(segments[j].data[1])
            maxY = np.max(segments[j].data[1])

        #If Slope is 0, line is horizontal
        #If Slope is infinity or -infinity, line is vertical
        #If Slope is >0 positive correlation
        #If Slope is <0 negative correlation
        #If Slope is >1, >0 Increase in Y faster than Increase in X
        #If Slope is <1, >0 Increase in Y slower than Increase in X
        #If Slope is <-1, <0 Decrease in Y faster than Increase in X
        #If Slope is >-1, <0 Decrease in Y slower than Increase in X

        #If Slope is slow in Y, then X has higher covariance
        #If Slope is fast in Y, then X has lower covariance

        #If X has higher covariance, use X as cut-off point
        #If Y has higher covariance, use Y as cut-off point

    #Hypothesis 1
        # proposal: extend all lines by a scalar value to encourage intersection
        # result: poor, some lines that already intersect do not need additional reach
        #           some lines require larger reach still to make important intersections
        # conclusion: We require a dynamic value per line, based on context?
        #
            if(abs(segments[j].slope[0][0]) < 1):
                x1 = minX
                #x1 = minX - np.minimum(minX,3) (Hypo 1)
                x2 = maxX
                #x2 = maxX + np.minimum(edges.shape[0] - maxX,3) (Hypo 1)
                if(segments[j].slope[0] > 0):
                    y1 = (np.multiply(segments[j].slope[0][0], x1) + segments[j].intercept)
                    y2 = (np.multiply(segments[j].slope[0][0], x2) + segments[j].intercept)
                else:
                    y2 = (np.multiply(segments[j].slope[0][0], x1) + segments[j].intercept)
                    y1 = (np.multiply(segments[j].slope[0][0], x2) + segments[j].intercept)
            else:
                y1 = minY
                #y1 = minY - np.minimum(minY,3) (Hypo 1)
                y2 = maxY
                #y2 = maxY + np.minimum(edges.shape[1] - maxY,3) (Hypo 1)
                if(segments[j].slope[0] > 0):
                    x1 = (np.divide((y1 - segments[j].intercept),segments[j].slope[0][0]))
                    x2 = (np.divide((y2 - segments[j].intercept), segments[j].slope[0][0]))
                else:
                    x2 = (np.divide((y1 - segments[j].intercept),segments[j].slope[0][0]))
                    x1 = (np.divide((y2 - segments[j].intercept), segments[j].slope[0][0]))

            segments[j].min[0], segments[j].min[1] = x1, y1
            segments[j].max[0], segments[j].max[1] = x2, y2

        #We want to encourage lines to intersect
        #However, we don't want this to happen arbitrarily
    #Hypothesis 2
        #If a line can be extended to intersect another, within the bounds of the others data points
        #Then it should do so.
        for j in range(first, last):
            for k in range(first,last):
                if(j < k):
                    #Do these lines intersect?
                    if(segments[j].slope[0] == segments[k].slope[0]):
                        #They never intersect
                        intersect = False
                    else:
                        #They intersect at [x_cross, y_cross]
                        #a1x + b1 = a2x + b2
                        #(a1 - a2)x = (b2 - b1)
                        #x = (b2-b1)/(a1-a2)
                        x_cross = np.divide((segments[k].intercept - segments[j].intercept),\
                                        (segments[j].slope[0] - segments[k].slope[0]))
                        #y = ax + b
                        y_cross = np.multiply(segments[j].slope[0], x_cross) + segments[j].intercept

                        #Does line j intersect line k within the data segment?
                        if ((x_cross < segments[k].max[0]) & (x_cross > segments[k].min[0])):
                            # Yes
                            intersect = True
                            # Extend line j to intersect line k
                            if (x_cross > segments[j].max[0]):
                                segments[j].max[0] = x_cross
                                if segments[j].slope[0] >= 0:
                                    segments[j].max[1] = y_cross
                                else:
                                    segments[j].min[1] = y_cross
                            else:
                                if (x_cross < segments[j].min[0]):
                                    segments[j].min[0] = x_cross
                                    if segments[j].slope[0] >= 0:
                                        segments[j].min[1] = y_cross
                                    else:
                                        segments[j].max[1] = y_cross
                                #else:
                                #They already intersect
                        else:
                            #Does line k intersect line j within the data segment?
                            if ((x_cross < segments[j].max[0]) & (x_cross > segments[j].min[0])):
                                #Yes
                                intersect = True
                                #Extend line k to intersect line j
                                if(x_cross > segments[k].max[0]):
                                    segments[k].max[0] = x_cross
                                    if(segments[k].slope[0] >= 0):
                                        segments[k].max[1] = y_cross
                                    else:
                                        segments[k].min[1] = y_cross
                                else:
                                    if(x_cross < segments[k].min[0]):
                                        segments[k].min[0] = x_cross
                                        if(segments[k].slope[0] >= 0):
                                            segments[k].min[1] = y_cross
                                        else:
                                            segments[k].max[1] = y_cross                                    #else:
                                    #They already intersect
                            else:
                                intersect = False

        #y1 = (np.multiply(line.slope, minX) + line.intercept)[0][0]
        #a = np.divide(np.ones(len(line.slope)), line.slope)
        #b = y1 - np.multiply(a, minX)
        #x2 = np.divide(line.intercept - b, a - line.slope)
        #y2 = (line.slope * x2) + line.intercept

        #if x2 < minX:
        #    minX = x2
        #if y2 < minY:
        #    minY = y2

        #x1 = (np.divide((minY - line.intercept),line.slope))[0][0]
        #y1 = (np.multiply(line.slope, minX) + line.intercept)[0][0]
        #x2 = (np.divide((maxY - line.intercept), line.slope))[0][0]
        #y2 = (np.multiply(line.slope, maxX) + line.intercept)[0][0]
        #if(y1 > minY):
        #    y1 = minY

        #x1 = minX

        #y2 = (np.multiply(line.slope, maxX) + line.intercept)[0][0]
        #if(y2 < maxY):
        #    y2 = maxY
        #x2 = maxX

    for line in segments:
        # If negative correlation, then [minX, maxY], [maxX, minY]
        if (line.slope[0] > 0):
            plt.plot([line.min[1], line.max[1]], [line.min[0], line.max[0]], 'r-')
        else:
            plt.plot([line.min[1], line.max[1]], [line.max[0], line.min[0]], 'r-')
