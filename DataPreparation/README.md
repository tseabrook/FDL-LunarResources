# Data Preparation

The Data Preparation folder is split into folders with the following files:
- **LROC_NAC** - Contains scripts for Lunar Reconnaissance Orbiter Camera, Narrow Angle Camera images to:
  - [**NAC1_PDS_grab.py**] Download NAC images from a provided list of product IDs
  - [**NAC2_Resample.py**] Resample NAC images, so that they may be comparative to LOLA_DEM images.
  - [**NAC3_Cut_Tiles.py**] Cut tiles from large NAC images, to be used by crater classifiers
  - [**NAC4_Clean_Tiles.py**] Clean tiles, filtering for dark lines and misshapen images. 

- **LOLA_DEM** - Contains scripts for Lunar [Reconnaissance] Orbiter Laser Altimeter, Digital Elevation Model to:
  - [**img_split.py**] - Split South Polar Region into Large Tiles 
  - [**img_split_2_latlon.py**] - Split Large Tiles into Smaller 32x32 Tiles, to be used by crater classifiers

- **Annotations** - Contains scripts for converting crater annotations generated from LabelImg and RectLabel tools:
  - [**DEMRectLabelConvert.py**] - Converts RectLabel annotated DEM 32x32pixel Tiles into a PASCAL_VOC format annotated 3800x3800 Large Tile
  - [**NACRectLabelConvert.py**] - Converts a RectLabel NAC image annotation into PASCAL_VOC format 
  - [**cut_PASCAL_VOCs.py**] - Draws PASCAL_VOC annotations onto a source image

- **AlternativeUnused** - Contains some additional scripts that haven't been used in production:
  - [**ClipTest_1.py**] - Dietmar's tiling prototype, retaining metadata
  - [**coordinateTransform.py**] - cthorey's PDS Extractor script, potentially useful for converting x-y to lat-lon
  - [**imageCoordinateTest.py**] - Dietmar's .IMG -> .tif converter with metadata
  - [**img_slicer.m**] - A matlab script for creating tiles with metadata
  - [**lat_lon_coordinate_calculator.py**] - A local implementation of cthorey's PDS Extractor script, to convert coordinates
        
