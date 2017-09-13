#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

import glob, os
from shutil import copyfile

first_dir = '/Users/seabrook/Dropbox/fdl_lunar_nac/polar_negative/eleni/'
second_dir = '/Users/seabrook/Dropbox/fdl_lunar_nac/polar_negative/tim/'
filetype = '*.tif'

def merge_folders(first_dir, second_dir, filetype):
    first_files = glob.glob(first_dir + filetype)
    second_files = glob.glob(second_dir + filetype)

    first_file_names = []
    second_file_names = []
    for filename in first_files:
        first_file_names += [filename.split('/')[-1]]
    for filename in second_files:
        second_file_names += [filename.split('/')[-1]]

    in_first = set(first_file_names)
    in_second = set(second_file_names)
    in_second_but_not_in_first = in_second - in_first
    in_first_but_not_in_second = in_first - in_second

    shared = in_first - in_first_but_not_in_second

    shared_filenames = list(shared)

    first_dir_name = first_dir.split('/')[-2]
    second_dir_name = second_dir.split('/')[-2]
    shared_dir_name = first_dir_name + '-' + second_dir_name

    #load root directory from first_dir location
    rootDir = first_dir.split(first_dir_name)[0]

    #initialise and/or create folder for images in both first_dir and second_dir
    shared_dir = rootDir + shared_dir_name +'/'
    if not os.path.isdir(shared_dir):
        os.makedirs(shared_dir)

    #initialise and/or create folder for images only in first_dir
    first_only_dir = shared_dir + first_dir_name  +'/'
    if not os.path.isdir(first_only_dir):
        os.makedirs(first_only_dir)

    # Move in_first_but_not_in_second to shared_dir/first_dir_name
    for filename in list(in_first_but_not_in_second):
        copyfile(first_dir + filename, first_only_dir + filename)

    #initialise and/or create folder for images only in second_dir
    second_only_dir = shared_dir + second_dir_name  +'/'
    if not os.path.isdir(second_only_dir):
        os.makedirs(second_only_dir)

    # Move in_second_but_not_in_first to shared_dir/second_dir_name
    for filename in list(in_second_but_not_in_first):
            copyfile(second_dir + filename, second_only_dir + filename)

    # Move shared_filenames to shared_dir
    for filename in list(shared):
        copyfile(first_dir + filename, shared_dir + filename)

merge_folders(first_dir, second_dir, '*.tif')