#Written by Dietmar Backes
#djbackes76@gmail.com

'''  Prototype for image cropping / cooky cut '''

try:
	import numpy
	#import pylab
	#import matplotlib.pyplot
	import os, sys
	#from osgeo import gdal, gdalnumeric, ogr, osr
	import gdal, gdalnumeric, ogr, osr
	import image  #,ImageDraw
    #gdal.UseExceptions()

except ImportError as err:
	print("ERROR: required python libraries are not prpoerly installed")
	print(err)
  	quit()


''' Parameter  '''
#---------------------------------------------------

#Inputfiles
Path = 'D:/1/Tiles/LOLA_DEM20/'
InputImage = 'lola_80s_20m.tif'
OutputPostfix = 'lola_20m.tif'
Shapefile = 'E:/1/LROC_5TO20KM_CRATERS_SPOLE.SHP'

#Outputs
MetaDATA = 'Metadata_lola_20m.txt'

#Control Parameter
RasterFormat = 'GTiff'
PixelRes = 20
VectorFormat = 'ESRI Shapefile'
BW = 0.30   #Buffer Width in %
#---------------------------------------------------


''' Open datasets '''

Raster = gdal.Open(Path+InputImage, gdal.GA_ReadOnly)
Projection = Raster.GetProjectionRef()

VectorDriver = ogr.GetDriverByName(VectorFormat)
VectorDataset = VectorDriver.Open(Shapefile, 0) # 0=Read-only, 1=Read-Write
layer = VectorDataset.GetLayer()

layerDefinition = layer.GetLayerDefn()


print "Name  -  Type  Width  Precision"
for i in range(layerDefinition.GetFieldCount()):
    fieldName =  layerDefinition.GetFieldDefn(i).GetName()
    fieldTypeCode = layerDefinition.GetFieldDefn(i).GetType()
    fieldType = layerDefinition.GetFieldDefn(i).GetFieldTypeName(fieldTypeCode)
    fieldWidth = layerDefinition.GetFieldDefn(i).GetWidth()
    GetPrecision = layerDefinition.GetFieldDefn(i).GetPrecision()
    print fieldName + " - " + fieldType+ " " + str(fieldWidth) + " " + str(GetPrecision)

FeatureCount = layer.GetFeatureCount()
print("Feature Count:",FeatureCount)

''' Clipping - Iterate through the shapefile features '''

#Initialise
Count_in = 0
Count_out = 0
output_file = open(Path+MetaDATA,'w')
output_file.write('No,Diam_km,Lat,Lon,Min_X,Min_Y,Max_X,Max_Y\n')



#Clip each valuable feature
for feature in layer:

    # Use Crater tile
    geom = feature.GetGeometryRef()
    minX, maxX, minY, maxY = geom.GetEnvelope()  # Get bounding box of the shapefile feature

    if ((minX >= -304000) & (maxX <= 304000) & (minY <= 304000) & (maxY >= -304000)):

        Count_out += 1
        #print("Processing feature " + str(Count_in) + " of " + str(FeatureCount) + "...")

        # create Buffer arond crater
        BufferX = abs((maxX-minX)*BW)
        BufferY = abs((maxY-minY)*BW)
        CutMinX = minX - BufferX
        CutMaxX = maxX + BufferX
        CutMinY = minY - BufferX
        CutMaxY = maxY + BufferX

        # Create Crater Tile as GTIFF + Meta Record

        #print (feature.GetField('Diam_km'))

        # Meta record
        line = '%i,%.3f,%.5f,%.5f,%.1f, %.1f, %.1f, %.1f\n' % (Count_out, feature.GetField('Diam_km'), feature.GetField('x_coord'), feature.GetField('y_coord'), minX, minY, maxX, maxY)

        output_file.write(line)
        # Raster
        OutTileName = Path+str(Count_out)+OutputPostfix
        OutTile = gdal.Warp(OutTileName, Raster, format=RasterFormat, outputBounds=[CutMinX, CutMinY, CutMaxX, CutMaxY], xRes=PixelRes, yRes=PixelRes, dstSRS=Projection, resampleAlg=gdal.GRA_NearestNeighbour, options=['COMPRESS=DEFLATE'])
        OutTile = None # Close dataset

    Count_in += 1
# Close datasets
Raster = None
VectorDataset.Destroy()
print("Done.")
