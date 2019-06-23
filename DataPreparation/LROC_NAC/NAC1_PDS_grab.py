#/usr/local/bin/python3

#Updated by Adam Lesnikowski
#Written by Timothy Seabrook
#timothy.seabrook@cs.ox.ac.uk

import urllib.request
import csv
from lxml import html
import requests
import numpy as np
import threading
import sys
import os
from IPython import embed

sys.path.append('../../Miscellanious/')

from timerInformation import timerInformation
from timeit import default_timer as timer

#PDS_GRAB
#This script reads a .txt file containing a list of desired NAC images
#The .txt file should contain more than zero rows with the first column containing NAC product IDs.


#https://stackoverflow.com/questions/16989647/importing-large-tab-delimited-txt-file-into-python

thisDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(thisDir, os.pardir, os.pardir)
dataDir = os.path.join(rootDir, 'Data')
NACDir = os.path.join(dataDir, 'LROC_NAC', 'South_Pole', 'Images')

if(not os.path.isdir(NACDir)): #SUBJECT TO RACE CONDITION
    os.makedirs(NACDir)

filename = os.path.join(dataDir,'P26_0-18000.txt')
filename = os.path.join(dataDir, 'own.txt')
file_object  = open(filename, 'r')

threads = []
d = []
with open(filename,'rb') as source:
    for line in source:
        try:
            #print('line is {}'.format(line))
            fields = line.decode('UTF-8').split('\t')
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        d.append(fields)

print('d is {}'.format(d))

perc_complete = 0  # Initialise percentage complete for timer
time_elapsed = 0  # Initialise time elapsed for timer
start = timer()  # Initialise timer

def save_image_from_url(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path) 
    except(IndexError):
        print('IndexError raised. Skipping saving this image...')

for i in range(1,len(d)):

    product_id = d[i][0]
    filetype = '.jpg' #'.jpeg' #'.tif'

    print('Searching for CDR Product: ' + product_id)

    save_path = os.path.join(NACDir,product_id+filetype)
    if(os.path.isfile(save_path)):
        print('Found CDR product at: '+save_path)
    else:
        # BEGIN TIMER UPDATE SECTION
        new_perc_complete = np.floor_divide(i * 100, len(d))  # update percentage complete integer
        if new_perc_complete > perc_complete:  # if integer is higher than last
            perc_complete = new_perc_complete  # then update
            start, time_elapsed = timerInformation('PDS NAC Grab', perc_complete, start, time_elapsed)  # and print ET
        # END TIMER UPDATE SECTION

        print('Connecting to http//moon.asu.edu')
        productURL = 'http://moon.asu.edu/planetview/inst/lroc/'+product_id
        print("productURL is {}".format(productURL))
        page = requests.get(productURL)
        print('Connected!')

        tree = html.fromstring(page.content)

        if filetype == '.tif' or filetype == '.tiff':
            downloadURL = tree.xpath('//*[@id="browseformats"]/a[4]')
        elif filetype == '.jpg' or filetype == '.jpeg':
            downloadURL = tree.xpath('//*[@id="browseformats"]/a[3]')
        else: 
            raise Exception('Unknown download file type. Try again please.')

        print('downloadURL is {}'.format(downloadURL))
        
        try:
            print('Identified image download URL: {}'.format(downloadURL[0].attrib['href']))
        except(IndexError):
            print('downloadURL print not possible')

        print('Downloading...')
        #urllib.retreive(
        try:
            save_image_from_url(downloadURL[0].attrib['href'], save_path)
        except(IndexError):
            print('An IndexError occurred. Skipping downloading this image data.')

        #embed()

        #t = threading.Thread(target=saveImage, args=(downloadURL[0].attrib['href'], save_path))
        #t.start()
        #threads.append(t)
        #map(lambda t: t.join(), threads)

        print("CDR Product downloaded: "+save_path)
