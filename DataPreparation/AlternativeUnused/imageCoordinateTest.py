#Written by Dietmar Backes
#djbackes76@gmail.com

#import shapefile
#import cv2
#http://pvl.readthedocs.io/en/latest/
#I decided to use PVL because there was better examples to be found in:
#https://github.com/cthorey/pdsimage/blob/master/pdsimage/PDS_Extractor.py

import pvl
from pvl import load as load_label

from osgeo import ogr, gdal


#NACFilenameIMG = 'imgs/M108898482RC.IMG'
NACFilenameIMG = 'PDS_FILES/LROC_NAC/M1236656672LC.IMG'
TIFFilenameIMG = 'imgs/M108898482RC.tif'
GA_ReadOnly = True

''' Open IMG images from PDS '''

dataset = gdal.Open(NACFilenameIMG, GA_ReadOnly)
if dataset is None:
    print ('could not read file')

print 'Driver: ', dataset.GetDriver().ShortName,'/', \
      dataset.GetDriver().LongName
print 'Size is ',dataset.RasterXSize,'x',dataset.RasterYSize, \
      'x',dataset.RasterCount
print 'Projection is ',dataset.GetProjection()

geotransform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

if not geotransform is None:
    print 'Origin = (',geotransform[0], ',',geotransform[3],')'
    print 'Pixel Size = (',geotransform[1], ',',geotransform[5],')'

label = pvl.load(NACFilenameIMG)

''' Write PDS-IMG images to GeoTIFF '''

format = "GTiff"
driver = gdal.GetDriverByName( format )
metadata = driver.GetMetadata()
if metadata.has_key(gdal.DCAP_CREATE) \
   and metadata[gdal.DCAP_CREATE] == 'YES':
    print 'Driver %s supports Create() method.' % format
if metadata.has_key(gdal.DCAP_CREATECOPY) \
   and metadata[gdal.DCAP_CREATECOPY] == 'YES':
    print 'Driver %s supports CreateCopy() method.' % format

src_filename = NACFilenameIMG
dst_filename = NACFilenameTIFF
src_ds = gdal.Open( src_filename )
dst_ds = driver.CreateCopy( dst_filename, src_ds, 0,
                            [ 'TILED=YES', 'COMPRESS=PACKBITS' ] )
# Once we're done, close properly the dataset
dst_ds = None
src_ds = None
#filename = 'M1236656672LC'
#fid = fopen(filename, 'rb')