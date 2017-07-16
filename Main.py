from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import glymur
import random
from skimage import data, io, filters
import os.path


def generateIlluminationMap(latitude, longitude, slopeMap):
    "This function generates an illumination map corresponding to the optical map provided."
    print("I'd like you to generate an illumination map for me.")
    w=slopeMap.shape[1]
    h=slopeMap.shape[0]

    illuminationMap = np.zeros((h,w))

    # PROTOTYPE: Sun is on western horizon
    for x in range(w):
        for y in range(h):
            if slopeMap[y][x][0][1] <= 0: #If the western pixel does not incline, then the pixel is illuminated
                illuminationMap[y][x] = 1

    #The following line doesn't work, but maybe there's a way.
    #illuminationMap[np.nonzero(slopeMap[:][:][1][0] >0)] = 1
    return illuminationMap

def generateDTEMap(latitude, longitude, slopeMap):
    "This function generates a DTE comms map corresponding to the slope map provided"
    print("I'd like you to generate a DTE comm map for me.")
    w=slopeMap.shape[1]
    h=slopeMap.shape[0]

    DTEMap = np.zeros((h,w))
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
            illuminationMapFilename = opticalMapFilename[0:opticalMapFilename.index('.')] + '_illum.npy'
            if os.path.isfile(illuminationMapFilename) is True:
                print("Illumination file found: ", illuminationMapFilename)
                self.illuminationMap = np.int_(np.load(illuminationMapFilename))
            else:
                print("Illumination file not found, generating illumination map")
                self.illuminationMap = np.int_(generateIlluminationMap(latitude, longitude, self.slopeMap))
                np.save(illuminationMapFilename, self.illuminationMap)
        else:
            self.illuminationMap = np.load(illuminationMapFilename)
        if (DTEMapFilename is None):
            print("No DTE map provided")
            print("Searching for matching DTE file")
            DTEMapFilename = opticalMapFilename[0:opticalMapFilename.index('.')] + '_DTE.npy'
            if os.path.isfile(DTEMapFilename) is True:
                print("DTE file found: ", DTEMapFilename)
                self.DTEMap = np.int_(np.load(DTEMapFilename))
            else:
                print("DTE file not found, generating DTE map")
                self.DTEMap = np.int_(generateDTEMap(latitude, longitude, self.slopeMap))
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

def main():
    #main initialises by loading relevant maps from database
    maps = [
            #TraverseMap('imgs/SLDEM2015_256_0N_60N_000_120.jp2',None,None,None,None)
            TraverseMap('imgs/prototype_plato_crater.jpeg', None, None, None, None)
        ]
#    agents =
    map = random.choice(maps)

    print("Searching for precomputed Region Map")
    regionMapFilename = 'imgs/regionMap1.npy'
    if os.path.isfile(regionMapFilename) is True:
        print("Region file found: ", regionMapFilename)
        regionMap = np.int_(np.load(regionMapFilename))
    else:
        print("Region file not found, generating Region map")
        regionMap, regionCounts = np.int_(connectedComponents(map))
        np.save(regionMapFilename, regionMap)
        np.save('imgs/regionMap1_counts.npy', regionCounts)

    agent = Agent(map)
    print(map)
    print(agent)

def timerInformation(jobname, percentage, start, elapsed):
    end = timer()
    elapsed += end - start
    start = timer()
    secs_remaining = ((elapsed * 100) / percentage) - elapsed
    hours_remaining = np.floor_divide(secs_remaining, 3600)
    secs_remaining = np.mod(secs_remaining, 3600)
    mins_remaining = np.floor_divide(secs_remaining, 60)
    secs_remaining = np.floor_divide(np.mod(secs_remaining, 60), 1)
    s = jobname+' is '+repr(percentage)+'% Complete. Estimated Time Remaining: '
    if hours_remaining > 0:
        s += repr(np.int_(hours_remaining)) + 'h '
    if mins_remaining > 0:
        s += repr(np.int_(mins_remaining)) + 'm '
    s += repr(np.int(secs_remaining)) + 's.'
    s += ' Time Elapsed: ' + repr(np.int_(np.floor_divide(elapsed, 60))) + 'mins.'
    print(s)
    return start, elapsed

def connectedComponents(map, roverMaxSlope=15):
    # Algorithm 1: Connected Components
    # Initialize all cells in Map to unlabeled2:
    #    while num(unlabeled) > 0
    #        cseed = next_unlabeled
    #        Setlabel(cseed) <- uniqueLabel
    #        C <- FLOODFILL3D(cseed)
    #        Setlabel(c) forall c in C
    #    end
    # functionFLOODFILL3D(seed)
    #    return the set C of all cells connected to seed
    # end
    w = map.img.shape[1]
    h = map.img.shape[0]
    labels = np.ones((h,w)) * -1
    candidates = np.nonzero(map.illuminationMap & map.DTEMap)
    labels[candidates[0],candidates[1]] = 0

    nextLabel = 0
    checkList = []
    seeds = np.where(labels == 0)
    num_total = seeds[0].size
    label_count = [0] #First element counts number of unilluminated and non-DTE pixels
    num_complete = 0
    perc_complete = 0
    time_elapsed = 0
    start = timer()
    while(seeds[0].size > 0):
        nextLabel += 1 #Begin a new label class
        label_count.append(0) #For each new label, begin a new count
        y = seeds[0][0]
        x = seeds[1][0]
        if(map.illuminationMap[y,x] == 1 & map.DTEMap[y,x] == 1): #If this pixel satisfies illumination and DTE
            labels[y,x] = nextLabel #Then begin the new group (else add to -1 pile)
            if checkList.__len__() == 0:
                checkList = [[y, x]]
            else:
                checkList = checkList.append([y,x]) #Add pixel to checklist

            while checkList.__len__() > 0: #checklist
                y, x = checkList.pop() #Take last pixel from checklist, to find qualifying neighbours
                num_complete += 1
                label_count[nextLabel] += 1
                #TIMER SECTION
                new_perc_complete = np.floor_divide(num_complete*100,num_total)
                if new_perc_complete > perc_complete:
                    perc_complete = new_perc_complete
                    #end = timer()
                    start, time_elapsed = timerInformation('Flood Fill', perc_complete, start, time_elapsed)
                    #start = timer()
                #END TIMER SECTION

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
                for i in range(xmin, xmax):
                    for j in range(ymin, ymax):
                        if (j != 1) ^ (i != 1):
                            if labels[y+j-1,x+i-1] == 0:
                                if np.absolute(map.slopeMap[y][x][i][j]) <= roverMaxSlope:
                                    if(map.illuminationMap[y+j-1,x+i-1] == 1) & (map.DTEMap[y+j-1,x+i-1] == 1):
                                        labels[y+j-1,x+i-1] = nextLabel
                                        checkList.append([y+j-1,x+i-1])
            seeds = np.where(labels == 0)
        else:
            labels[y,x] = -1
            num_remaining += 1
            label_count[0] += 1
            seeds = np.where(labels == 0)

    return labels, label_count


if __name__ == '__main__':
    main()