# Import library
from __future__ import print_function
import numpy as np
#import pandas as pd
import os
import sys
from distutils.util import strtobool
# Library to help read header in binary WAC FILE
import pvl
from pvl import load as load_label

class BinaryTable(object):
    """ Class to read image binary file from the LRO experiment
        For the moment, it can gather information about the topography
        (from LRO LOLA experiment) and texture (from the LRO WAC
        experiment). More information about the Lunar Reconnaissance
        Orbiter mission (LRO) can be found `here`_
        LRO LOLA - Informations can be found at `LRO/LOLA website`_.
        In particular, the header, located in  a separate file .LBL file,
        contains all the informations.
        LROC WAC - Informations can be found at `LROC/WAC website`_.
        In particular, HEADER in the binary file that contain all the information -
        Read with the module `pvl module`_ for informations about how the header is
        extracted directly from the file.
        This class has a method able to download the images,
        though it might be better to download them before
        as it takes a lot of time, especially for large resolution.
        Both are NASA PDS FILE - Meaning, they are binary table whose format depends on
        the file. All the information can be found in the Header whose
        reference are above. Line usualy index latitude while sample on the line
        refers to longitude.
        THIS CLASS SUPPORT ONLY CYLINDRICAL PROJECTION FOR THE MOMENT.
        PROJECTION : [WAC : 'EQUIRECTANGULAR', LOLA : '"SIMPLE"']
        FURTHER WORK IS NEEDED FOR IT TO BECOMES MORE GENERAL.
        Args:
            fname (str): Name of the image.
            path_pdsfiles (Optional[str]): Path where the pds files are stored.
                Defaults, the path is set to the folder ``PDS_FILES`` next to
                the module files where the library is install.
                See ``defaut_pdsfile`` variable of the class
        Attributes:
            fname (str): Name of the image.
            path_pdsfiles (str): path where the pds files are stored.
            lolapath (str): path for LOLA images
            wacpath (str): path for WAC images
            grid (str): WAC or LOLA
            img (str): name of the image
            lbl (str): name of the lbl file, where information are stored. Empty for WAC.
        Note:
            It is important to respect the structure of the PDS_FILES folder. It
            should contain 2 subfolder called ``LOLA`` and ``LROC_WAC`` where the
            corresponding images should be download.
            I also integrate all the specification of the image contained in
            the header or the .LBL file as attribute of the class. However,
            the list is long and I do not introduce them into the
            documentation. See the file directly for details.
            The abreaviations correspond to:
            - **LRO** Lunar Reconnaissance Orbiter
            - **LOLA** Lunar Orbiter Laser Altimeter
            - **LROC** Lunar Reconnaissance Orbiter Camera
            - **WAC** Wide Angle Camera
        .. _here:
            http://www.nasa.gov/mission_pages/LRO/spacecraft/#.VpOMDpMrKL4
        .. _LRO/LOLA website:
            http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/aareadme.txt
        .. _LROC/WAC website:
            http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/AAREADME.TXT
        .. _pvl module:
            http://pvl.readthedocs.org/en/latest/
        """

    #Will add files to PDS_Files sub-folder of current folder location
    default_pdsfile = os.path.join(
        '/'.join(os.path.abspath(__file__).split('/')[:-1]), 'PDS_FILES')

    def __init__(self, fname, path_pdsfile=default_pdsfile):
        self.fname = fname.upper()
        self.grid = 'NAC'

        self.path_pdsfiles = path_pdsfile
        if not os.path.isdir(self.path_pdsfiles):
            print('% s: The directory where PDS_FILES should be does\
                          not exist. Creation of the directory.' % (self.path_pdsfiles))
            try:
                os.mkdir(self.path_pdsfiles)
            except:
                raise BaseException('The creation of %s abort.\
                                            Might be a permission problem if you\
                                            do not provide any path and you install\
                                            the library in a read - only directory. Please\
                                            provide a valid path.')
        elif not os.access(self.path_pdsfiles, os.W_OK):
            raise BaseException("% s: The directory where the PDS file are\
                                        is read only. It might be the defaut\
                                        path if you install in a directory\
                                        without any rights. Please change it\
                                        for a path with more permission to\
                                        store PDS_FILES" % (self.path_pdsfiles))
        else:
            print('PDS FILES used are in: %s' % (self.path_pdsfiles))

        #NAC sub-folder of PDS_FILES folder
        self.nacpath = os.path.join(self.path_pdsfiles, 'LROC_NAC')
        if not os.path.isdir(self.nacpath):
            print('Creating a directory NAC_LROC under %s' % (self.nacpath))
            os.mkdir(self.nacpath)

        self.img = os.path.join(self.nacpath, self.fname + '.IMG')
        self.MAP_PROJECTION_TYPE = 'POLAR_STEREOGRAPHIC'
        self._load_info_lbl()

    def _load_info_lbl(image):
        """ Load info on the image
        Note:
            If the image is from LOLA, the .LBL is parsed and the
            information is returned.
            If the image is from NAC, the .IMG file is parsed using
            the library `pvl`_ which provide nice method to extract
            the information in the header of the image.
        .. _pvl: http://pvl.readthedocs.org/en/latest/
        """
        label = load_label(image.img)
        for key, val in label.iteritems():
            if type(val) == pvl._collections.PVLObject:
                for key, value in val.iteritems():
                    try:
                        setattr(image, key, value.value)
                    except:
                        setattr(image, key, value)
            else:
                setattr(image, key, val)
        image.start_byte = image.RECORD_BYTES
        image.bytesize = 4
        image.projection = str(label['IMAGE_MAP_PROJECTION'][
                                  'MAP_PROJECTION_TYPE'])
        image.dtype = np.float32

def long_id(self, sample, line):
    ''' Return the corresponding longitude
    Args:
        sample (int): sample number on a line
    Returns:
        Correponding longidude in degree
    '''

    #   North Polar Stereographic
    #   Lon = LonP + atan(x/(-y))
    #   South Polar Stereographic
    #   Lon = LonP + atan(x/y)
    #   x = (Sample - S0 - 1) * Scale
    #   y = (1 - L0 - Line) * Scale

    y = (1 + self.LINE_PROJECTION_OFFSET - line) * self.MAP_SCALE * 1e-3
    x = (sample - self.SAMPLE_PROJECTION_OFFSET - 1) * self.MAP_SCALE * 1e-3
    lonP = self.CENTER_LONGITUDE
    lon = lonP + np.arctan(np.divide(x,y))
    return lon

def lat_id(self, line):
    ''' Return the corresponding latitude
    Args:
        line (int): Line number
    Returns:
        Corresponding latitude in degree
    '''
    P = np.sqrt(np.power(x,2) + np.power(y,2))
    R = 1737.4
    C = 2*np.arctan(np.multiply(np.divide(P,2), R))
    latP = -90
    y = (1 + self.LINE_PROJECTION_OFFSET - line) * self.MAP_SCALE * 1e-3
    #Lat = ArcSin[Cos(c)*Sin(LatP) + y*Sin(C)*CosLatP/P]
    lat = np.arcsin(np.multiply(np.cos(C), np.sin(latP)) + (np.multiply(np.multiply(y,np.sin(C)), np.divide(np.cos(latP), P))) )
    return lat

filename = 'M1236656412CC'
BinaryTable(filename)
print('done')