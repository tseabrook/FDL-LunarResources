import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from scipy.sparse import csr_matrix
from graphCycles import Graph
from split_and_merge import split_and_merge, line_of_best_fit, \
        line_distances, point_distance, attach_lines, \
        generate_line_ends, connect_lines

import glymur

def drawGraph(segments):
    edges = []
    for i in range(segments.shape[0]):
        if(segments[i].start_connect >= 0):
            edges.append((segments[i].start_connect, i))
        if (segments[i].end_connect >= 0):
            edges.append((i, segments[i].end_connect))

    return nx.DiGraph(list(set(edges)))

def checkCycles(segments):
    edges = []
    for i in range(segments.shape[0]):
        if (segments[i].start_connect >= 0):
            edges.append((segments[i].start_connect, i))
        if (segments[i].end_connect >= 0):
            edges.append((i, segments[i].end_connect))
    num_edges = len(edges)
    g = Graph(num_edges)
    for edge in edges:
        g.addEdge(edge[0], edge[1])

    return g.isCyclic()

def findCycles(G):
    try:
        cycles = list(nx.find_cycle(G, orientation='ignore'))
    except:
        pass
        cycles = []
    return cycles


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
hypothesis = 4


for n in range(2):
    #curr_filename = filename+str(n+1)+'.jp2'
    curr_filename = 'imgs/testdata/test_img.jp2'
    # Load picture and detect edges
    image = glymur.Jp2k(curr_filename)[:]
    # Low threshold and High threshold represent number of pixels that may be skipped to make a line [4, 60 seems good]
    # Sigma represents the width of the guassian smoothing kernel [3 seems good]
    edges = canny(image, sigma=3, low_threshold=4, high_threshold=50)
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
            segments = np.hstack((segments, split_and_merge(lines[i], 0.5)))
            segmentParent[i] = segments.size


    cm = plt.get_cmap('gist_rainbow')
    f1, axarr = plt.subplots(ncols=2, nrows=1)
    axarr[0].imshow(edges, cmap=plt.cm.gray)
    axarr[1].imshow(image, cmap=plt.cm.gray)
    axarr[1].set_color_cycle([cm(1. * i / 20) for i in range(20)])
    for i in range(1,len(lines)):
        y, x = lines[i]
        axarr[1].scatter(x, y, alpha=0.8, edgecolors='none', s=1)

    f2, axarr = plt.subplots(ncols=2, nrows=1)
    axarr[0].imshow(image, cmap=plt.cm.gray)
    axarr[1].imshow(image, cmap=plt.cm.gray)
    #For every grouped line
    for i in range(1,len(lines)):
        first = segmentParent[i-1]
        last = segmentParent[i]

        #For every segment of line
        plt.axes(axarr[0])
        for j in range(first,last):
            generate_line_ends(segments[j])
            plt.plot([segments[j].start[1], segments[j].end[1]], [segments[j].start[0], segments[j].end[0]], 'r-')

    #Hypothesis 1
        # proposal: extend all lines by a scalar value to encourage intersection
        # result: poor, some lines that already intersect do not need additional reach
        #           some lines require larger reach still to make important intersections
        # conclusion: We require a dynamic value per line, based on context?
        #
    #Hypothesis 2
        # proposal: where two lines can intersect if extended by max((end-mean/2),max_extend)
        #           they should be
        # result: decent, large lines extend too far, most 'easy' craters get captured.
        # conclusion: distance between ends of lines is probably better than distance to intersection
        #
        #If a line can be extended to intersect another, within the bounds of the others data points
        #Then it should do so.
        #Max extension (in x) permissible for each of two lines to intersect
##############################################################################
        if(hypothesis == 2):
            max_extend = 5
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

                            #Check that intersection point lies within bounds of map
                            if((x_cross > 0) & (x_cross < edges.shape[0]) & (y_cross > 0) & (y_cross < edges.shape[1])):
                                #If x_cross is outside of segment j's maximal bounds
                                if (x_cross > segments[j].max[0]):
                                    #check that x_cross is close enough to j to warrant intersection
                                    if ((x_cross - segments[j].max[0]) < np.maximum(np.multiply(0.5,(
                                                np.max(segments[j].data[0]) - segments[j].mean[0])),max_extend)):
                                        #If x_cross is outside of segment k's maximals bounds
                                        if (x_cross > segments[k].max[0]):
                                            # check that x_cross is close enough to k to warrant intersection
                                            if ((x_cross - segments[k].max[0]) < np.maximum(np.multiply(0.5, (
                                                    np.max(segments[k].data[0]) - segments[k].mean[0])), max_extend)):
                                                #If it is, update k(max)
                                                segments[k].max[0] = x_cross
                                                if (segments[k].slope[0] >= 0):
                                                    segments[k].max[1] = y_cross
                                                else:
                                                    segments[k].min[1] = y_cross
                                                #update j(max)
                                                segments[j].max[0] = x_cross
                                                if segments[j].slope[0] >= 0:
                                                    segments[j].max[1] = y_cross
                                                else:
                                                    segments[j].min[1] = y_cross
                                        else:
                                            # If x_cross is outside of segment k's minimal bounds
                                            if (x_cross < segments[k].min[0]):
                                                # check that x_cross is close enough to k to warrant intersection
                                                if ((segments[k].min[0] - x_cross) < np.maximum(np.multiply(0.5, (
                                                        segments[k].mean[0] - np.min(segments[k].data[0]))),max_extend)):
                                                    # If it is, update k(min)
                                                    segments[k].min[0] = x_cross
                                                    if (segments[k].slope[0] >= 0):
                                                        segments[k].min[1] = y_cross
                                                    else:
                                                        segments[k].max[1] = y_cross
                                                    #update j(max)
                                                    segments[j].max[0] = x_cross
                                                    if segments[j].slope[0] >= 0:
                                                        segments[j].max[1] = y_cross
                                                    else:
                                                        segments[j].min[1] = y_cross
                                            else: #x_cross is within bounds of k
                                                # update j(max)
                                                segments[j].max[0] = x_cross
                                                if segments[j].slope[0] >= 0:
                                                    segments[j].max[1] = y_cross
                                                else:
                                                    segments[j].min[1] = y_cross
                                else:
                                    # If x_cross is outside of segment j's minimal bounds
                                    if (x_cross < segments[j].min[0]):
                                        # check that x_cross is close enough to j to warrant intersection
                                        if((segments[j].min[0] - x_cross) < np.maximum(np.multiply(0.5,(
                                                    segments[j].mean[0] - np.min(segments[j].data[0]))),max_extend)):
                                            # If x_cross is outside of segment k's maximal bounds
                                            if (x_cross > segments[k].max[0]):
                                                # check that x_cross is close enough to k to warrant intersection
                                                if ((x_cross - segments[k].max[0]) < np.maximum(np.multiply(0.5,(
                                                        np.max(segments[k].data[0]) - segments[k].mean[0])),max_extend)):
                                                    # If it is, update k(max)
                                                    segments[k].max[0] = x_cross
                                                    if (segments[k].slope[0] >= 0):
                                                        segments[k].max[1] = y_cross
                                                    else:
                                                        segments[k].min[1] = y_cross
                                                    # update j(min)
                                                    segments[j].min[0] = x_cross
                                                    if segments[j].slope[0] >= 0:
                                                        segments[j].min[1] = y_cross
                                                    else:
                                                        segments[j].max[1] = y_cross
                                            else:
                                                # If x_cross is outside of segment k's minimal bounds
                                                if (x_cross < segments[k].min[0]):
                                                    # check that x_cross is close enough to k to warrant intersection
                                                    if ((segments[k].min[0] - x_cross) < np.maximum(np.multiply(0.5, (
                                                                segments[k].mean[0] - np.min(segments[k].data[0]))), max_extend)):
                                                        # If it is, update k(min)
                                                        segments[k].min[0] = x_cross
                                                        if (segments[k].slope[0] >= 0):
                                                            segments[k].min[1] = y_cross
                                                        else:
                                                            segments[k].max[1] = y_cross
                                                        # update j(min)
                                                        segments[j].min[0] = x_cross
                                                        if segments[j].slope[0] >= 0:
                                                            segments[j].min[1] = y_cross
                                                        else:
                                                            segments[j].max[1] = y_cross
                                                else: #x_cross is within bounds of k
                                                    # update j(max)
                                                    segments[j].min[0] = x_cross
                                                    if segments[j].slope[0] >= 0:
                                                        segments[j].min[1] = y_cross
                                                    else:
                                                        segments[j].max[1] = y_cross
                                    else: #x_cross is within bounds of j
                                        # If x_cross is outside of segment k's maximals bounds
                                        if (x_cross > segments[k].max[0]):
                                            # check that x_cross is close enough to k to warrant intersection
                                            if ((x_cross - segments[k].max[0]) < np.maximum(np.multiply(0.5,
                                                (np.max(segments[k].data[0]) - segments[k].mean[0])), max_extend)):
                                                # If it is, update k(max)
                                                segments[k].max[0] = x_cross
                                                if (segments[k].slope[0] >= 0):
                                                    segments[k].max[1] = y_cross
                                                else:
                                                    segments[k].min[1] = y_cross
                                        else:
                                            # If x_cross is outside of segment k's minimal bounds
                                            if (x_cross < segments[k].min[0]):
                                                # check that x_cross is close enough to k to warrant intersection
                                                if ((segments[k].min[0] - x_cross) < np.maximum(np.multiply(0.5, (
                                                            segments[k].mean[0] - np.min(segments[k].data[0]))), max_extend)):
                                                    # If it is, update k(min)
                                                    segments[k].min[0] = x_cross
                                                    if (segments[k].slope[0] >= 0):
                                                        segments[k].min[1] = y_cross
                                                    else:
                                                        segments[k].max[1] = y_cross
                                            #else:  # x_cross is within bounds of k
##############################################################################
        # Hypothesis 3
        # proposal: Connecting the ends of lines will provide more sensible connections
        #           than connecting intersections
        # result: Compact groups, lots of unnecessary crossing lines.
        # conclusion: Most lines only need to connect once at each end

        if(hypothesis == 3):
            max_extend = 6
            changeFlag = True
            connected = np.zeros((last - first, last - first), dtype=bool)
            while(changeFlag):
                changeFlag = False
                for j in range(first, last):
                    for k in range(first,last):
                        if(j < k):
                            if(connected[j-first,k-first] == False):
                                #First, do these lines already intersect?
                                if (segments[j].slope[0] == segments[k].slope[0]):
                                    # They never intersect
                                    intersect = False
                                else:
                                    x_cross = np.divide((segments[k].intercept[0] - segments[j].intercept[0]),
                                                        (segments[j].slope[0] - segments[k].slope[0]))
                                    # y = ax + b
                                    y_cross = np.multiply(segments[j].slope[0], x_cross) + segments[j].intercept[0]
                                    intersect = False
                                    #if((x_cross > segments[k].min[0]) & (x_cross > segments[j].min[0])
                                    #        & (x_cross < segments[k].max[0]) & (x_cross < segments[j].max[0])):
                                    #    intersect = True
                                    #    connected[j-first,k-first] = True
                                    #    connected[k-first,j-first] = True
                                if(intersect == False):
                                    #Are the ends of these lines close together?
                                    distance = np.zeros(4)
                                    #min -> min
                                    distance[0] = np.sqrt(np.sum((np.power(segments[j].start[0] - segments[k].start[0],2),
                                        np.power((segments[j].start[1] - segments[k].start[1]), 2))))
                                    #min -> max
                                    distance[1] = np.sqrt(np.sum((np.power((segments[j].start[0] - segments[k].end[0]),2),
                                        np.power((segments[j].start[1] - segments[k].end[1]), 2))))
                                    #max -> min
                                    distance[2] = np.sqrt(np.sum((np.power((segments[j].end[0] - segments[k].start[0]),2),
                                        np.power((segments[j].end[1] - segments[k].start[1]), 2))))
                                    #max -> max
                                    distance[3] = np.sqrt(np.sum((np.power((segments[j].end[0] - segments[k].end[0]),2),
                                        np.power((segments[j].end[1] - segments[k].end[1]), 2))))
                                    ind = np.argmin(distance)
                                    if distance[ind] < max_extend:
                                        if(distance[ind] == 0):
                                            connected[j - first, k - first] = True
                                            connected[k - first, j - first] = True
                                        else:
                                            changeFlag = True
                                            switcher = {
                                                0: [[segments[j].start[0], segments[j].start[1]], [segments[k].start[0], segments[k].start[1]]],
                                                1: [[segments[j].start[0], segments[j].start[1]], [segments[k].end[0], segments[k].end[1]]],
                                                2: [[segments[j].end[0], segments[j].end[1]], [segments[k].start[0], segments[k].start[1]]],
                                                3: [[segments[j].end[0], segments[j].end[1]], [segments[k].end[0], segments[k].end[1]]],
                                            }
                                            data = switcher.get(ind)
                                            connected[j - first, k - first] = True
                                            connected[k - first, j - first] = True
                                            segments = np.insert(segments, last, line_of_best_fit(data))
                                            segments[last].start = [data[0][0], data[0][1]]
                                            segments[last].end = [data[1][0], data[1][1]]
                                            segmentParent[i:] = segmentParent[i:]+1
##############################################################################
        # Hypothesis 4
        # proposal: A greedy search for new end-of-line connections up to a maximum of 1 connection at each end
        #           Followed by a greedy search for loose end-of-line connections
        # result: Much tidier groups, though lines appear jittery.
        # conclusion: It might be better to move nodes rather than draw new edges.

        if (hypothesis == 4):
            big_number = 9999999999999
            max_extend = 6
            connected_lines = np.zeros(last - first,dtype=bool)
            connected = np.zeros((last-first, last-first),dtype=bool)
            nodes = []
            #for j in range(first, last):
            #    for k in range(first, last):
            #        if (j < k):
                        # First, do these lines already intersect?
                        #if (segments[j].slope[0] == segments[k].slope[0]):
                            # They never intersect, but could connect
                         #   if(segments[j].intercept[0] == segments[k].intercept[0]):
                                #They are on the same line
                                #Only need to check x value equality, since lines are parallel
                          #      if(((segments[j].start[0] >= segments[k].start[0])
                           #         & (segments[j].start[0] <= segments[k].end[0]))
                            #        ^ ((segments[j].start[0] >= segments[k].end[0])
                             #       & (segments[j].start[0] <= segments[k].start[0]))):
                             ##       segments[j].start_connect = k
                             #       connected[j-first, k-first] = True
                             ##       connected[k-first, j-first] = True
                             #   if (((segments[j].end[0] >= segments[k].start[0])
                             #       & (segments[j].end[0] <= segments[k].end[0]))
                             #       ^ ((segments[j].end[0] >= segments[k].end[0])
                             #       & (segments[j].end[0] <= segments[k].start[0]))):
                             #       segments[j].end_connect = k
                             #       connected[j-first, k-first] = True
                             #       connected[k-first, j-first] = True
                             #   if (((segments[k].start[0] >= segments[j].start[0])
                             #       & (segments[k].start[0] <= segments[j].end[0]))
                             #       ^ ((segments[k].start[0] >= segments[j].end[0])
                             #       & (segments[k].start[0] <= segments[j].start[0]))):
                             #       segments[k].start_connect = j
                             #       connected[j-first, k-first] = True
                             #       connected[k-first, j-first] = True
                             #   if (((segments[k].end[0] >= segments[j].start[0])
                             ###       & (segments[k].end[0] <= segments[j].end[0]))
                             #       ^ ((segments[k].end[0] >= segments[j].end[0])
                             #       & (segments[k].end[0] <= segments[j].start[0]))):
                             #       segments[k].end_connect = j
                             #       connected[j-first, k-first] = True
                             #       connected[k-first, j-first] = True#

                                # The next pair of conditions should NEVER occur
                                # However, the check has been included for sanity
                             #   if((segments[j].end_connect == k)
                             #       & (segments[j].start_connect == k)):
                             #           #(Line j < Line k) ^ (Line j = Line k)
                             #           np.delete(segments, j, 0)
                             #           last = last - 1
                             #           segmentParent[i:] = segmentParent[i:] + -1
                             #           np.delete(connected_lines, j-first, 0)
                             #           np.delete(connected, j-first, 0)
                             #           np.delete(connected, j-first, 1)
                             #   else:
                             #       if ((segments[k].end_connect == j)
                             #           & (segments[k].start_connect == j)):
                             #           #Line k < Line j
                             #           np.delete(segments, k, 0)
                             #           last = last - 1
                             #           segmentParent[i:] = segmentParent[i:] + -1
                             #           np.delete(connected_lines, k-first, 0)
                             #           np.delete(connected, k-first, 0)
                             #           np.delete(connected, k-first, 1)

                        #The lines are not parallel, continue intersection check
                        #else:
                            # x = (b2 - b1)/(a1 - a2)
                        #    x_cross = np.rint(np.divide(
                        #        (segments[k].intercept[0] - segments[j].intercept[0]),
                        #        (segments[j].slope[0] - segments[k].slope[0])))

                            #This introduces bugs due to errors introduced through division
                            #Rounding could help, but the direction of rounding would need to be know

                            #if ((x_cross >= segments[k].min[0]) & (x_cross >= segments[j].min[0])
                            #        & (x_cross <= segments[k].max[0]) & (x_cross <= segments[j].max[0])):
                            #    #Lines intersect!
                            #    #But where...?
                            #    if(abs(segments[k].end[0] - x_cross) < abs(segments[k].start[0] - x_cross)):
                            #        segments[k].end_connect = j
                            #    else:
                            #        segments[k].start_connect = j
                            #    if(abs(segments[j].end[0] - x_cross) < abs(segments[j].start[0] - x_cross)):
                            #        segments[j].end_connect = k
                            #    else:
                            #        segments[j].start_connect = k
                            #    connected[j-first,k-first] = True
                            #    connected[k-first,j-first] = True

            #If start and end of line is connected, then do not connect them again
            #for j in range(first, last):
            #    if ((segments[j].start_connect >= 0) &  (segments[j].end_connect >= 0)):
            #        connected_lines[j-first] = True

            #Find lines that haven't been fully connected yet
            unconnected = np.where(connected_lines == False)[0]+first
            num_lines = unconnected.shape[0]

            #Build adjacency matrix for lines that haven't been connected
            line_adjacency = np.zeros((num_lines, num_lines,4), dtype=float)

            #For lines that haven't been fully connected...
            ##########Calculate line end distances
            for j in range(num_lines):
                for k in range(num_lines):
                    if j < k:
                        #Not considering joined pairs of partially connected lines
                        if(connected[j,k] == True):
                            line_adjacency[j,k,0] = big_number
                            line_adjacency[j,k,1] = big_number
                            line_adjacency[j,k,2] = big_number
                            line_adjacency[j,k,3] = big_number
                        else:
                            #Measure the distance between the ends of the lines
                            #Ensure that lines are unconnected before measuring distance
                            # start -> start
                            line_adjacency[j,k,:] = line_distances(segments[unconnected[j]],segments[unconnected[k]])
                    else:
                        if(j == k):
                            line_adjacency[j, k, 0] = big_number
                            line_adjacency[j, k, 1] = big_number
                            line_adjacency[j, k, 2] = big_number
                            line_adjacency[j, k, 3] = big_number
                        else:
                            # If line has already been processed, copy distance values
                            line_adjacency[j, k,0] = line_adjacency[k, j,0]
                            line_adjacency[j, k,1] = line_adjacency[k, j,2]
                            line_adjacency[j, k,2] = line_adjacency[k, j,1]
                            line_adjacency[j, k,3] = line_adjacency[k, j,3]

            connect_flag = True
            l = 0

            #Whilst there are still partially connected lines less than [max_extend] distance apart
            while(connect_flag == True):
                #Find the shortest distance (greedy strategy)
                # argmin gives flatIndex,
                #   use unravel_index with array shape to return 3d index
                #If the shortest distance is acceptable
                if line_adjacency.size == 0:
                    connect_flag = False
                else:
                    j, k, l = np.unravel_index(np.argmin(line_adjacency), line_adjacency.shape)
                    if line_adjacency[j,k,l] < max_extend:

                        if(line_adjacency[j,k,l] == 0):
                            attach_lines(segments[unconnected[j]], segments[unconnected[k]], l)
                            connected[k, j] = True
                            connected[j, k] = True
                            line_adjacency[j, k, :] = big_number
                            line_adjacency[k, j, :] = big_number

                        else:
                            #Create a new line to bridge the distance
                            segments = np.insert(segments, last,
                                    connect_lines(segments[unconnected[j]], segments[unconnected[k]], l))

                            segmentParent[i:] = segmentParent[i:] + 1
                            connected = np.hstack((connected, np.zeros((last-first, 1), dtype=bool)))
                            connected = np.vstack((connected, np.zeros((1,last-first+1), dtype=bool)))
                            connected[k, last-first] = True
                            connected[j, last-first] = True
                            connected[last-first, k] = True
                            connected[last-first, j] = True
                            connected[k,j] = True
                            connected[j,k] = True
                            line_adjacency[j, k, :] = big_number
                            line_adjacency[k, j, :] = big_number

                            #Adjacency switcher is used to select relevant line_adjacency values
                            #For each 'connection made type' row:
                            #First values identify connections types that line1 can no longer make
                            #Second values identify connections types that line2 can no longer make
                            #Third values identify connections types that j can no longer receive
                            #Fourth values identify connections types that k can no longer receive
                            adjacency_switcher = {
                                0: [[0, 1],[0, 1],[0, 2],[0, 2]], #Type start->start
                                1: [[0, 1],[2, 3],[0, 2],[1, 3]], #Type start->end
                                2: [[2, 3],[0, 1],[1, 3],[0, 2]], #Type end->start
                                3: [[2, 3],[2, 3],[1, 3],[1, 3]], #Type end->end
                            }
                            inds = adjacency_switcher[l]
                            line_adjacency[j,:,inds[0]] = big_number
                            line_adjacency[k,:,inds[1]] = big_number
                            line_adjacency[:,j,inds[2]] = big_number
                            line_adjacency[:,k,inds[3]] = big_number

                            last = last + 1

                        diff = 0
                        if ((segments[unconnected[j]].start_connect >= 0) & (segments[unconnected[j]].end_connect >= 0)):
                            connected_lines[j] = True
                            unconnected = np.delete(unconnected, j, 0)
                            line_adjacency = np.delete(line_adjacency, j, 0)
                            line_adjacency = np.delete(line_adjacency, j, 1)
                            num_lines = num_lines - 1
                            if k > j:
                                diff = 1

                        if ((segments[unconnected[k-diff]].start_connect >= 0) & (segments[unconnected[k-diff]].end_connect >= 0)):
                            connected_lines[k] = True
                            unconnected = np.delete(unconnected, k-diff, 0)
                            line_adjacency = np.delete(line_adjacency, k-diff, 0)
                            line_adjacency = np.delete(line_adjacency, k-diff, 1)
                            num_lines = num_lines - 1

                    else:
                        connect_flag = False

            #Now there are only partially connected lines remaining
            #We should see if these can connect to any nearby lines
            num_remain = unconnected.shape[0]
            #unconnected have been being deleted upon full-connection during previous step
            line_adjacency = np.zeros((last-first, 4))
            #max_extend = 10
            for j in range(num_remain):
                for k in range(last-first):
                    #Cannot connect to self
                    if(unconnected[j] == k+first):
                        line_adjacency[k, :] = big_number
                    else:
                        #Cannot reconnect over previously connections
                        if(connected[unconnected[j]-first,k] == True):
                            line_adjacency[k,:] = big_number
                        else:
                            #Measure distance to all other ends of lines
                            if(segments[unconnected[j]].start_connect < 0):
                                line_adjacency[k, 0] = point_distance(segments[unconnected[j]].start,segments[k+first].start)
                                line_adjacency[k, 1] = point_distance(segments[unconnected[j]].start,segments[k+first].end)
                            else:
                                line_adjacency[k, 0] = big_number
                                line_adjacency[k, 1] = big_number
                            if(segments[unconnected[j]].end_connect < 0):
                                line_adjacency[k, 2] = point_distance(segments[unconnected[j]].end,segments[k+first].start)
                                line_adjacency[k, 3] = point_distance(segments[unconnected[j]].end,segments[k+first].end)
                            else:
                                line_adjacency[k, 2] = big_number
                                line_adjacency[k, 3] = big_number
#                            line_distances(segments[unconnected[j]],segments[k+first])

                k, l = np.unravel_index(np.argmin(line_adjacency), line_adjacency.shape)

                #If shortest distance is below threshold, make connection
                if line_adjacency[k,l] < max_extend:
                    if (line_adjacency[k,l] == 0): #If shortest distance indicates prior connection, form connection formally
                        connected[unconnected[j] - first, k] = True
                        connected[k, unconnected[j] - first] = True
                        attach_lines(segments[unconnected[j]], segments[k+first], l)
                    else:
                        changeFlag = True

                        segments = np.insert(segments, last,
                                 connect_lines(segments[unconnected[j]], segments[k+first], l))

                        connected[unconnected[j] - first, k] = True
                        connected[k, unconnected[j] - first] = True
                        segmentParent[i:] = segmentParent[i:] + 1

                        connected = np.hstack((connected, np.zeros((last - first, 1), dtype=bool)))
                        connected = np.vstack((connected, np.zeros((1, last - first + 1), dtype=bool)))
                        connected[k, last-first] = True
                        connected[unconnected[j]-first, last-first] = True
                        connected[last-first, k] = True
                        connected[last-first, unconnected[j]-first] = True
                        line_adjacency[k, :] = big_number
                        if((k+first) in unconnected):
                            line_adjacency[np.where(unconnected==(k+first))[0]] = big_number
                        line_adjacency = np.vstack((line_adjacency, np.multiply(np.ones((1,4)),big_number)))

                        last = last + 1

        #print(checkCycles(segments[first:last]))
        plt.axes(axarr[1])
        for m in range(first,last):
            plt.plot([segments[m].start[1], segments[m].end[1]], [segments[m].start[0], segments[m].end[0]], 'r-')

        print('done')
        #cycles = findCycles(drawGraph(segments[first:last]))
        #if (len(cycles) > 0):
        #    print(cycles)

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

    #for line in segments:
        # If negative correlation, then [minX, maxY], [maxX, minY]
    #    plt.plot([line.start[1], line.end[1]], [line.start[0], line.end[0]], 'r-')
        #if (line.slope[0] > 0):
        #    plt.plot([line.min[1], line.max[1]], [line.min[0], line.max[0]], 'r-')
        #else:
        #    plt.plot([line.min[1], line.max[1]], [line.max[0], line.min[0]], 'r-')

print("end")