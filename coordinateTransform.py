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

    default_pdsfile = os.path.join(
        '/'.join(os.path.abspath(__file__).split('/')[:-1]), 'PDS_FILES')

    def __init__(self, fname, path_pdsfile=default_pdsfile):

        self.fname = fname.upper()
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

        self.lolapath = os.path.join(self.path_pdsfiles, 'LOLA')
        self.wacpath = os.path.join(self.path_pdsfiles, 'LROC_WAC')
        self.nacpath = os.path.join(self.path_pdsfiles, 'LROC_NAC')
        if not os.path.isdir(self.lolapath):
            print('Creating a directory LOLA under %s' % (self.lolapath))
            os.mkdir(self.lolapath)
        if not os.path.isdir(self.wacpath):
            print('Creating a directory WAC_LROC under %s' % (self.wacpath))
            os.mkdir(self.wacpath)
        if not os.path.isdir(self.nacpath):
            print('Creating a directory NAC_LROC under %s' % (self.nacpath))
            os.mkdir(self.nacpath)
        self._category()
        self._maybe_download()
        self._load_info_lbl()

        assert self.MAP_PROJECTION_TYPE in [
            '"SIMPLE', 'EQUIRECTANGULAR'], "Only cylindrical projection is possible - %s NOT IMPLEMENTED" % (self.MAP_PROJECTION_TYPE)



    def _category(self):
        """ Type of the image: LOLA or WAC
        Note: Specify the attribute ``grid``, ``img`` and ``lbl`
        """

        if self.fname.split('_')[0] == 'NAC':
            self.grid = 'NAC'
            self.img = os.path.join(self.nacpath, self.fname + '.IMG')
            self.lbl = ''
        elif self.fname.split('_')[0] == 'WAC':
            self.grid = 'WAC'
            self.img = os.path.join(self.wacpath, self.fname + '.IMG')
            self.lbl = ''
        elif self.fname.split('_')[0] == 'LDEM':
            self.grid = 'LOLA'
            self.img = os.path.join(self.lolapath, self.fname + '.IMG')
            self.lbl = os.path.join(self.lolapath, self.fname + '.LBL')
        else:
            raise ValueError("%s : This type of image is not recognized. Possible\
                             images are from %s only" % (self.fname, ', '.join(('WAC', 'LOLA'))))

    def _load_info_lbl(self):
        """ Load info on the image
        Note:
            If the image is from LOLA, the .LBL is parsed and the
            information is returned.
            If the image is from NAC, the .IMG file is parsed using
            the library `pvl`_ which provide nice method to extract
            the information in the header of the image.
        .. _pvl: http://pvl.readthedocs.org/en/latest/
        """

        if self.grid == 'WAC':
            label = load_label(self.img)
            for key, val in label.iteritems():
                if type(val) == pvl._collections.PVLObject:
                    for key, value in val.iteritems():
                        try:
                            setattr(self, key, value.value)
                        except:
                            setattr(self, key, value)
                else:
                    setattr(self, key, val)
            self.start_byte = self.RECORD_BYTES
            self.bytesize = 4
            self.projection = str(label['IMAGE_MAP_PROJECTION'][
                                      'MAP_PROJECTION_TYPE'])
            self.dtype = np.float32
        elif self.grid == 'NAC':
            label = load_label(self.img)
            for key, val in label.iteritems():
                if type(val) == pvl._collections.PVLObject:
                    for key, value in val.iteritems():
                        try:
                            setattr(self, key, value.value)
                        except:
                            setattr(self, key, value)
                else:
                    setattr(self, key, val)
            self.start_byte = self.RECORD_BYTES
            self.bytesize = 4
            self.projection = str(label['IMAGE_MAP_PROJECTION'][
                                      'MAP_PROJECTION_TYPE'])
            self.dtype = np.float32
        else:
            with open(self.lbl, 'r') as f:
                for line in f:
                    attr = [f.strip() for f in line.split('=')]
                    if len(attr) == 2:
                        setattr(self, attr[0], attr[1].split(' ')[0])
            self.start_byte = 0
            self.bytesize = 2
            self.projection = ''
            self.dtype = np.int16

def long_id(self, sample):
    ''' Return the corresponding longitude
    Args:
        sample (int): sample number on a line
    Returns:
        Correponding longidude in degree
    '''
    if self.grid == 'WAC':
        lon = self.CENTER_LONGITUDE + (sample - self.SAMPLE_PROJECTION_OFFSET - 1) \
                                      * self.MAP_SCALE * 1e-3 / (
                                      self.A_AXIS_RADIUS * np.cos(self.CENTER_LATITUDE * np.pi / 180.0))
        return lon * 180 / np.pi
    else:
        lon = float(self.CENTER_LONGITUDE) + \
              (sample - float(self.SAMPLE_PROJECTION_OFFSET) - 1) \
              / float(self.MAP_RESOLUTION)
        return lon

def lat_id(self, line):
    ''' Return the corresponding latitude
    Args:
        line (int): Line number
    Returns:
        Correponding latitude in degree
    '''
    if self.grid == 'WAC':
        lat = ((1 + self.LINE_PROJECTION_OFFSET - line) *
               self.MAP_SCALE * 1e-3 / self.A_AXIS_RADIUS)
        return lat * 180 / np.pi
    else:
        lat = float(self.CENTER_LATITUDE) - \
              (line - float(self.LINE_PROJECTION_OFFSET) - 1) \
              / float(self.MAP_RESOLUTION)
        return lat


#fname (str): Name of the image.
#        path_pdsfiles (str): path where the pds files are stored.
#        lolapath (str): path for LOLA images
#        wacpath (str): path for WAC images
#        grid (str): WAC or LOLA
#        img (str): name of the image
#        lbl (str): name of the lbl file, where information are stored. Empty for WAC.
