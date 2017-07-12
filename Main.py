import matplotlib.pyplot as plt
import numpy as np
import glymur
import random

def generateIlluminationMap(opticalMapFilename):
    "This function generates an illumination map corresponding to the optical map provided."
    print("I'd like you to generate an illumination map for me for optical map.", opticalMapFilename)
    return None

def generateDTEMap(slopeMapFilename):
    "This function generates a DTE comms map corresponding to the slope map provided"
    print("I'd like you to generate a DTE comm map for me from.", slopeMapFilename)
    return None

def generateSlopeMap(opticalMapFilename):
    "This function generates a slope map corresponding to the optical map provided"
    print("I'd like you to generate a slope map for me from ", opticalMapFilename)
    return None

class Map:
    #Map is a combination of several layers:
    #[latitude]
    #[longitude]
    #[opticalMapFilename] Optical Map
    #[slopeMapFilename] Slope Map
    #[illuminationMapFilename] Illumination Map
    #[iceMapFilename] Subsurface Ice Map
    #[DTEMapFilename] Direct to Earth (DTE) Communications Map
    def __init__(self, latitude = None, longitude = None, opticalMapFilename=None, slopeMapFilename=None, illuminationMapFilename=None,
                 iceMapFilename=None, DTEMapFilename=None):
        #assert type(opticalMapFilename) is str,\
        #         'argument to Map initialisation must be a string filename'
        #self.filename = filename
        self.longitude = longitude
        self.latitude = latitude

        if (opticalMapFilename is None):
            print("No optical map provided")
            self.img = open('blank.jpg', 'r')
        else:
            self.jp2 = glymur.Jp2k(opticalMapFilename)
            self.img = self.jp2[:]
        if (slopeMapFilename is None):
            print("No slope map provided")
            self.slopeMap = generateSlopeMap(opticalMapFilename)
        else:
            self.slopeMap = open(slopeMapFilename, 'r')
        if (illuminationMapFilename is None):
            print("No illumination map provided")
            self.illuminationMap = generateIlluminationMap(self.slopeMap)
        else:
            self.illuminationMap = open(illuminationMapFilename, 'r')
        if (iceMapFilename is None):
            print("No ice map provided")
            self.iceMap = None
        else:
            self.iceMap = open(iceMapFilename, 'r')
        if (DTEMapFilename is None):
            self.DTEMap = generateDTEMap(self.slopeMap)
        else:
            self.DTEMap = open(DTEMapFilename, 'r')

    def __str__(self):
        return self.print_mapName()

    def print_mapName(self):
        return 'Map selected: \'{filename}\''.format(filename=self.filename)

class Agent:
    def __init__(self, map):
        assert type(map) is Map,\
                 'argument to Agent initialisation must be a Map'
        self.initialState = [random.randrange(1, map.img.shape[0]), random.randrange(1, map.img.shape[1])]

    def __str__(self):
        return self.print_agentInit()

    def print_agentInit(self):
        return 'Agent initialised at location: {initialState}'.format(initialState=self.initialState)

def main():

    maps = [
            Map('imgs/SLDEM2015_256_0N_60N_000_120.jp2',None,None,None,None)
        ]
    map = random.choice(maps)
    agent = Agent(map)
    print(map)
    print(agent)

if __name__ == '__main__':
    main()
    



# Parameters.

#Use Glymur to open JPEG2000 image
#Load image contents into 'image'
##thumbnail = jp2[::2, ::2]
#shape = thumbnail.shape
#print(shape)
#width =
#height =#

#print(thumbnail)

