from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import glymur
import random
from skimage import data, io, filters, img_as_uint, exposure
import gc
from scipy.sparse import csr_matrix
import os.path
import matplotlib
from collections import namedtuple
from operator import itemgetter
from pprint import pformat

def generateIlluminationMap(latitude, longitude, slopeMap, opticalMap):
    "This function generates an illumination map corresponding to the optical map provided."
    print("I'd like you to generate an illumination map for me.")

    # PROTOTYPE A: Sun is on western horizon
   # w=slopeMap.shape[1]
   # h=slopeMap.shape[0]
   # illuminationMap = np.zeros((h,w))
   # for x in range(w):
   #     for y in range(h):
   #         if slopeMap[y][x][0][1] <= 0: #If the western pixel does not incline, then the pixel is illuminated
   #             illuminationMap[y][x] = 1

    # PROTOTYPE B: Illumination matching colour of photo
    #               dims linearly until complete darkness
    w=opticalMap.shape[1]
    h=opticalMap.shape[0]
    illuminationMap = np.zeros((8,h,w), dtype=bool)
    for i in range(8):
        illuminationMap[i] = opticalMap > ((i*30)+10)

    #
    #The following line doesn't work, but maybe there's a way.
    #illuminationMap[np.nonzero(slopeMap[:][:][1][0] >0)] = 1
    return illuminationMap

def generateDTEMap(latitude, longitude, slopeMap):
    "This function generates a DTE comms map corresponding to the slope map provided"
    print("I'd like you to generate a DTE comm map for me.")
    w=slopeMap.shape[1]
    h=slopeMap.shape[0]

    DTEMap = np.zeros((h,w), dtype=bool)
    # PROTOTYPE: Earth is on northern horizon
    for x in range(w):
        for y in range(h):
            if slopeMap[y][x][1][0] <= 0: #If the northern pixel does not incline, then the pixel is DTE
                DTEMap[y][x] = 1
    return DTEMap

def generateSlopeMap(terrainMapFilename):
    "This function generates a slope map corresponding to the optical map provided"
    #print("Generating a slope map from: ", terrainMapFilename)
    #terrainMap = open(terrainMapFilename,'r')
    terrainMap = terrainMapFilename
    w = terrainMap.shape[1]
    h = terrainMap.shape[0]
    #Each x,y coordinate of the optical map is converted into an 8-directional slope map (3x3 list)
    #NOTE: THIS IS NOT OPTIMAL
    #((x*2-1),(y*2-1)) map is optimal - Eleni, this is your job now :)

    slopeMap = np.zeros((h,w,3,3))

    def diff(x,y):
        return y-x

    for x in range(w): #For all pixel columns
        for y in range(h): #For all pixel rows
            if x > 0:
                xmin = 0
                if x < (w - 1): #middle column
                    xmax = 3
                else: #rightmost column
                    xmax = 2
            else: #leftmost column
                xmax = 3
                xmin = 1
            if y > 0:
                ymin = 0
                if y < (h - 1): #middle row
                    ymax = 3
                else: #bottom row
                    ymax = 2
            else: #top row
                ymax = 3
                ymin = 1

            for i in range(xmin,xmax):
                for j in range(ymin,ymax):
                    slopeMap[y][x][i][j] = diff(terrainMap[y][x],terrainMap[y-1+j][x-1+i])

    return slopeMap

#class TerrainModel:
#    "Objects of class TerrainModel contain a 2 dimensional altimeter map." \
#    "Examples of altimeter datasets include: LOLA, MOLA and GLAS."
#    def __init__(self, terrainModelFilename):
#        self.altimeterMap = open(terrainModelFilename, 'r')

class TraverseMap:
    #TraverseMap is a combination of several layers taken as inputs:
    #[latitude] (Float)
    #[longitude] (Float)
    #[opticalMapFilename] Optical Map
    #[slopeMapFilename] Slope Map (Float[x,y])
    #[roughnessMapFilename] Roughness Map (Float[x,y])
    #[iceMapFilename] Subsurface Ice Depth Stability Map (Float[x,y])
    #[illuminationMapFilename] Illumination Map (Boolean[x,y,t])
    #[DTEMapFilename] Direct to Earth (DTE) Communications Map (Boolean[x,y,t])
    #
    #Traverse Map Attributes:
    #[width] number of pixels along the longitudinal horizontal axis of combined map layers
    #[height] number of pixels along the latitudinal vertical axis of combined map layers
    #[length] number of samples along the temporal depth axis of combined map layers
    #[img] JPEG2000 Loaded from opticalMapFilename (Float[width,height])
    #[slopeMap] 2D Array of gradients, loaded from slopeMapFilename or calculated from associated TerrainModel
    #           (Float[width,height])
    #[roughnessMap] 2D Array of roughness metric, loaded from roughnessMapFilename
    #iceMap
    #illuminationMap
    #DTEMap

    def __init__(self, opticalMapFilename=None, slopeMapFilename=None, illuminationMapFilename=None,
                 roughnessMapFilename=None, iceMapFilename=None, DTEMapFilename=None,
                 latitude = None, longitude = None, ):
        #assert type(opticalMapFilename) is str,\
        #         'argument to Map initialisation must be a string filename'
        #self.filename = filename
        self.longitude = longitude
        self.latitude = latitude

        if (opticalMapFilename is None):
            print("No optical map provided")
            #self.img = open('blank.jpg', 'r')
        else:
            if opticalMapFilename.endswith('.jp2'):
                jp2 = glymur.Jp2k(opticalMapFilename)
                self.img = jp2[:]
            else:
                self.img = io.imread(opticalMapFilename)

        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

        # Slope map is taken from precomputed data source.
        if (slopeMapFilename is None):
            print("No slope map provided")
            print("Searching for matching slope file")
            slopeMapFilename = opticalMapFilename[0:opticalMapFilename.index('.')] + '_slope.npy'
            if os.path.isfile(slopeMapFilename) is True:
                print("Slope file found")
                self.slopeMap = np.load(slopeMapFilename)
            else:
                print("Slope file not found, generating slope map")
                self.slopeMap = generateSlopeMap(np.int_(self.img))
                np.save(slopeMapFilename, self.slopeMap)
            #a) PROTOTYPE: If slope map is not provided, generate one from optical map
            #b) ALPHA: If slope map is not provided generate one from TerrainModel

        else:
            self.slopeMap = np.load(slopeMapFilename)

        # Roughness Map is taken directly from data sources.
        if (roughnessMapFilename is None):
            print("No roughness map provided")
            #If roughness map is not provided, then generate a zero roughness grid.
            self.roughnessMap = [[0] * self.width for i in range(self.height)]
        else:
            self.roughnessMap = open(roughnessMapFilename, 'r')
        if (iceMapFilename is None):
            print("No ice map provided")
            #If ice map is not provided, then generate a zero ice grid
            self.iceMap = [[0] * self.width for i in range(self.height)]
        else:
            self.iceMap = open(iceMapFilename, 'r')
        if (illuminationMapFilename is None):
            print("No illumination map provided")
            print("Searching for matching illumination file")
            illuminationMapFilename = opticalMapFilename[0:opticalMapFilename.index('.')] + '_illumB.npy'
            if os.path.isfile(illuminationMapFilename) is True:
                print("Illumination file found: ", illuminationMapFilename)
                self.illuminationMap = np.load(illuminationMapFilename)
            else:
                print("Illumination file not found, generating illumination map")
                self.illuminationMap = generateIlluminationMap(latitude, longitude, self.slopeMap, self.img)
                np.save(illuminationMapFilename, self.illuminationMap)
        else:
            self.illuminationMap = np.load(illuminationMapFilename)
        if (DTEMapFilename is None):
            print("No DTE map provided")
            print("Searching for matching DTE file")
            DTEMapFilename = opticalMapFilename[0:opticalMapFilename.index('.')] + '_DTE.npy'
            if os.path.isfile(DTEMapFilename) is True:
                print("DTE file found: ", DTEMapFilename)
                self.DTEMap = np.load(DTEMapFilename)
            else:
                print("DTE file not found, generating DTE map")
                self.DTEMap = generateDTEMap(latitude, longitude, self.slopeMap)
                np.save(DTEMapFilename, self.DTEMap)
        else:
            self.DTEMap = np.load(DTEMapFilename)

    def __str__(self):
        return self.print_mapName()

    def print_mapName(self):
        return 'Region selected: \'{lat}, {lon}\''.format(lat=self.latitude, lon=self.longitude)

class Agent:
    #Agent operates on a Map
    #Agent has attributes:
    #Initial State (On map) (I)
    #Available Actions per State (S x A)
    #Transition Function (S x A x S)
    #Movement Speed (m/s)
    #Max Slope (dY/dX)
    def __init__(self, map, attributes=None):
        assert type(map) is TraverseMap,\
                 'argument to Agent initialisation must be a TraverseMap'
        #1a) Initial State is chosen randomly from the provided map tile
        #1b) Initial State is chosen as (one of) the lowest density qualifying position(s) on provided map tile
        self.initialState = [random.randrange(1, map.img.shape[0]), random.randrange(1, map.img.shape[1])]
        self.attributes = attributes
    def __str__(self):
        return self.print_agentInit()

    def print_agentInit(self):
        return 'Agent initialised at location: {initialState}'.format(initialState=self.initialState)


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def countPixelsPerRegion_fast(regionMap):
    #https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
    #MUCH faster than looping np.where (>10x)
    #Cannot handle negative values however, so those are removed beforehand
    indices = get_indices_sparse(regionMap[np.where(regionMap != -1)])
    counts = np.zeros((np.max(regionMap)+1))
    for i in range(np.max(regionMap)+1):
        counts[i] = indices[i][0].size
    return indices, counts



def timerInformation(jobname, percentage, start, elapsed):
    #timerInformation updates and prints [elapsed] times for [jobname] with an estimated time remaining
    end = timer() #Timestamp
    elapsed += end - start #Time since last start call to timer() is added to overall elapsed time
    start = timer() #New timer() is started
    secs_remaining = ((elapsed * 100) / percentage) - elapsed #calculating overall job time minus elapsed
    hours_remaining = np.floor_divide(secs_remaining, 3600)
    secs_remaining = np.mod(secs_remaining, 3600)
    mins_remaining = np.floor_divide(secs_remaining, 60)
    secs_remaining = np.floor_divide(np.mod(secs_remaining, 60), 1)
    s = jobname+' is '+repr(percentage)+'% Complete. Estimated Time Remaining: ' #Print job and percentage completed
    if hours_remaining > 0: #Conditionally print hours remaining
        s += repr(np.int_(hours_remaining)) + 'h '
    if mins_remaining > 0: #Conditionally print minutes remaining
        s += repr(np.int_(mins_remaining)) + 'm '
    s += repr(np.int(secs_remaining)) + 's.' #Print time remaining
    s += ' Time Elapsed: ' + repr(np.int_(np.floor_divide(elapsed, 60))) + 'mins.' #Print time elapsed
    print(s)
    return start, elapsed #Return newly initialised timer and current elapsed time

def illuminationDTEEligibility(illuminationMap, DTEMap):
    candidates = illuminationMap & DTEMap
    return np.nonzero(candidates)



def connectedComponents(map, roverMaxSlope=50):
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
    w, h, d = map.img.shape[1], map.img.shape[0], map.illuminationMap.shape[0] #Size of search area


    labels = np.zeros((d,h,w), dtype=np.uint32) #uint32 covers 0 to 4,294,967,295
    valid_pixels = np.zeros((d,h,w), dtype=bool) #initalise indicator for whether pixel should be searched over

    candidates = illuminationDTEEligibility(map.illuminationMap, map.DTEMap) #find all pixels that satisfy illumination and DTE
    valid_pixels[candidates[0],candidates[1],candidates[2]] = True

    #candidates = illuminationDTEEligibility(map.illuminationMap, map.DTEMap) #Pixels that satisfy illumination and DTE

    # NOT OPTIMAL - OPTIMAL REQUIRES EACH ROW TO BE NEW LABEL, EACH COL AS 1D INDEX
    #  #Become candidates for connected components search
    nextLabel = 0 #Region ID (0 means unlabelled)
    checkList = [] #Initialise checklist, contains pixels for neighbourhood traversability checks
    seeds = np.nonzero(valid_pixels) #Find all valid unlabelled pixels
    num_total = seeds[0].size #Count number of valid unlabelled pixels
    num_complete = 0 #Initialise counter
    next_ind = 0
    perc_complete = 0 #Initialise percentage complete for timer
    time_elapsed = 0 #Initialise time elapsed for timer
    start = timer() #Initialise timer
    #BEGIN CONNECTED COMPONENTS ALGORITHM

    while(num_complete < num_total):
        nextLabel += 1 #Increment label class ID
        z, y, x = seeds[0][next_ind], seeds[1][next_ind], seeds[2][next_ind]
        while(labels[z,y,x] != 0):
            next_ind += 1
            z, y, x = seeds[0][next_ind], seeds[1][next_ind], seeds[2][next_ind]

        labels[z,y,x] = nextLabel #Add next pixel to the new label class

        if checkList.__len__() == 0: #Create a list of pixels for FloodFill neighbour checking
            checkList = [[z, y, x]]
        else:
            checkList = checkList.append([z, y, x])

        #BEGIN FLOODFILL ALGORITHM
        while checkList.__len__() > 0: #Whilst there are qualifying pixels in this iteration of FloodFill
            z, y, x = checkList.pop() #Take pixel from checklist, to find qualifying neighbours
            num_complete += 1 #update count for timer

            # BEGIN TIMER UPDATE SECTION [Placement outside FloodFill improves speed (4x fewer divisions for 15 roverMaxSlope)
            #                               but reduces initial accuracy]
            new_perc_complete = np.floor_divide(num_complete * 100, num_total)  # update percentage complete integer
            if new_perc_complete > perc_complete:  # if integer is higher than last
                perc_complete = new_perc_complete  # then update
                start, time_elapsed = timerInformation('Flood Fill', perc_complete, start, time_elapsed)  # and print ET
                # END TIMER UPDATE SECTION

            #BEGIN LOCATION SPECIFIC NEIGHBOUR INDEXING
            if z > 0:
                zmin = -1
                if z < (d - 1):
                    zmax = 2
                else:
                    zmax = 1
            else:
                zmax = 2
                zmin = 0
            if x > 0:
                xmin = 0
                if x < (w - 1): #middle column
                    xmax = 3
                else: #rightmost column
                    xmax = 2
            else: #leftmost column
                xmax = 3
                xmin = 1
            if y > 0:
                ymin = 0
                if y < (h - 1): #middle row
                    ymax = 3
                else: #bottom row
                    ymax = 2
            else: #top row
                ymax = 3
                ymin = 1
            #END LOCATION SPECIFIC NEIGHBOUR INDEXING

            #BEGIN NEIGHBOUR TRAVERSABILITY CHECK
            for k in range(zmin, zmax):
                for i in range(xmin, xmax):
                    for j in range(ymin, ymax): #for all neighbouring pixels
                        if (((j == 1) & (i == 1) & (k == 0))!=True): #not including current pixel
                            if valid_pixels[z+k,y+j-1,x+i-1] == True: #and only considering unlabeled pixels
                                if np.absolute(map.slopeMap[y][x][i][j]) <= roverMaxSlope: #check if they can be reached from this pixel
                                    #if(map.illuminationMap[y+j-1,x+i-1] == 1) & (map.DTEMap[y+j-1,x+i-1] == 1): #not necessary, already checked in precompute
                                    labels[z+k,y+j-1,x+i-1] = nextLabel
                                    checkList.append([z+k,y+j-1,x+i-1])
            #END NEIGHBOUR TRAVERSABILITY CHECK
        #END FLOODFILL ALGORITHM
        #seeds = np.where(labels == 0) #Reset candidate seeds
    #END CONNECTED COMPONENTS ALGORITHM
    return labels #return labels and count


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


def kdtree(point_list, depth=0):
    try:
        k = len(point_list[0])  # assumes all points have the same dimension
    except IndexError as e:  # if not point_list:
        return None
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # Sort point list and choose median as pivot element
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2  # choose median

    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )

#
# def ocTree(points):
#
#     binDepths = [0] #Initialise
#
# def Astar(start, goal, illuminationMap):
#     d = illuminationMap.shape[0]
#     h = illuminationMap.shape[1]
#     w = illuminationMap.shape[2]
#     #The set of nodes already evaluated
#     closedSet = []
#
#     #The set of currently discovered nodes that are not evaluated yet
#     #Initially only the start node is known
#     openSet = [start]
#
#     #For each node, which node it can most efficiently be reached from
#     #If a node can be reached from many nodes, cameFrom will eventually
#     #contain the most efficient previous step
#     cameFrom = np.zeros((d,h,w), dType=np.uint32)
#
#     #For each node, the cost of getting from the start node to that node
#     gScore = np.ones((d,h,w), dType=np.int64) * -1
#
#     #The cost of going from start to start is zero
#     gScore[start] = 0
#
#     #For each node, the total cost of getting from the start node to the goal
#     #by passing by that node. THat value is partly known, partly heuristic
#     fScore := map with default value of Infinity
#
#     #For the first node, that value is completely heuristic.
#     fScore[start] := heuristic_cost_estimate(start,goal)
#
#     while openSet is not empty
#         current := the node in openSet having the lowest fScore[] value
#         if current = goal
#             return reconstruct_path(cameFrom, current)
#
#         openSet.Remove(current)
#         closedSet.Add(current)
#
#         for each neighbour of current
#             if neighbour in closedSet
#                 continue #Ignore the neighbour which is already evaluated
#
#             if neighbour not in openSet #Discover a new node
#                 openSet.Add(neighbour)
#
#             #The distance from start to a neighbour
#             tentative_gScore := gScore[current] + dist_between(current, neighbour)
#             if tentative_gScore >= gScore[neighbour]
#                 continue #This is not a better path.
#             #This path is the best until now. Record it!
#             cameFrom[neighbour] := current
#             gScore[neighbour] := tentative_gScore
#             fScore[neighbour] := gScore[neighbour] + heuristic_cost_estimate(neighbour, goal)
#     return failure
#
# def reconstruct_path(cameFrom, current):
#     total_path := [current]
#     while current in cameFrom.Keys:
#         current := cameFrom[current]
#         total_path.append(current)
#     return total_path

def main():
    #main initialises by loading relevant maps from database
    maps = [
            #TraverseMap('imgs/SLDEM2015_256_0N_60N_000_120.jp2',None,None,None,None)
            TraverseMap('imgs/prototype_plato_crater.jpeg', None, None, None, None)
        ]
#    agents =
    map = random.choice(maps)

    #kdmap.illuminationMap


    print("Searching for precomputed Region Map")
    regionMapFilename = 'imgs/regionMap_150-3D.npy'
    if os.path.isfile(regionMapFilename) is True:
        print("Region file found: ", regionMapFilename)
        regionMap = np.int_(np.load(regionMapFilename))
        regionCounts = np.load('imgs/regionMap_counts_150-3D.npy')
    else:
        print("Region file not found, generating Region map")
        regionMap = np.int_(connectedComponents(map))
        regionIndices, regionCounts = countPixelsPerRegion_fast(regionMap)
        np.save(regionMapFilename, regionMap)
        np.save('imgs/regionMap_counts_150-3D.npy', regionCounts)


    maxRegion = np.where(regionCounts == np.max(regionCounts))
    regionIndices = np.where(regionMap == maxRegion)
    #z_ind = regionIndices[0]
    y_ind = regionIndices[1]
    x_ind = regionIndices[2]

    #y_ind, x_ind = np.floor_divide(regionIndices[maxRegion][0], regionMap[0].size), np.modulus(regionIndices[maxRegion][0], regionMap[0].size)

    for i in range(regionMap.shape[0]):
        im = np.array([[0, 0, 0]], dtype='float64')
        im = np.matlib.repmat(im,regionMap[i].shape[0],regionMap[i].shape[1]).reshape(regionMap[i].shape[0],regionMap[i].shape[1],3)
        im[y_ind,x_ind,0] = 1
        im = img_as_uint(im)
        io.imsave('imgs/connectedRegion_8bit_'+i+'.png', im)

    #a0 = np.array([chr(0) + chr(0) + chr(0)])
    #data = np.matlib.repmat(a0, regionMap.shape[0], regionMap.shape[1])
    #im = Image.new("RGB", (regionMap.shape[0], regionMap.shape[1]), "black")
    #pix = im.load()
    #pix[y_ind, x_ind] = [chr(255) + chr(0) + chr(0)]
    #im = Image.frombytes("RGB", (regionMap.shape[0], regionMap.shape[1]), data)
    #im.

    agent = Agent(map)
    print(map)
    print(agent)


if __name__ == '__main__':
    main()




#### LEGACY CODE ####
#
#    def countPixelsPerRegion_simple(regionMap):
#        maxID = np.max(regionMap)
#        labelCount = np.zeros((maxID))  # Get number of connected regions
#        labelCount[0] = np.where(regionMap == -1)[0].size
#        for i in range(1, maxID + 1):
#            labelCount[i] = np.where(regionMap == i)[0].size  # Count number of pixels in connected region i
#        return labelCount