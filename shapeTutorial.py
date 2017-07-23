import shapefile

shapeFilename = 'imgs/craters/LROC_5TO20KM_CRATERS_SPOLE/LROC_5TO20KM_CRATERS_SPOLE.SHP'
sf = shapefile.Reader(shapeFilename)
sfShapes = sf.shapes()

numShapes = len(sfShapes)

for shape in sfShapes:
    bbox = shape.bbox #Bounding boxes in polar reference meters [e.g. meters from the poles]
    print(bbox)
    #Now we have the bbox and can grab the craters from a co-registered NAC or DEM. Ensuring that the coordinate systems are the same.
    #Meters -> Pixels
    #