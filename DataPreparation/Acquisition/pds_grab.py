import urllib2
import csv
from lxml import html
import requests
import numpy as np
import threading
from Main import timerInformation
from timeit import default_timer as timer

def saveImage(url, path):
    req = urllib2.Request(url)
    resp = urllib2.urlopen(req)
    imgdata = resp.read()
    with open(path, 'wb') as outfile:
        outfile.write(imgdata)

#https://stackoverflow.com/questions/16989647/importing-large-tab-delimited-txt-file-into-python
filename = '/Volumes/DATA DISK/PDS_FILES/LROC_NAC/P26_0-18000.txt'
file_object  = open(filename, 'r')

threads = []
d = []
with open(filename,'rb') as source:
    for line in source:
        fields = line.split('\t')
        d.append(fields)

perc_complete = 0  # Initialise percentage complete for timer
time_elapsed = 0  # Initialise time elapsed for timer
start = timer()  # Initialise timer

for i in range(1,len(d)):

    # BEGIN TIMER UPDATE SECTION
    new_perc_complete = np.floor_divide(i * 100, len(d))  # update percentage complete integer
    if new_perc_complete > perc_complete:  # if integer is higher than last
        perc_complete = new_perc_complete  # then update
        start, time_elapsed = timerInformation('PDS NAC Grab', perc_complete, start, time_elapsed)  # and print ET
    # END TIMER UPDATE SECTION

    product_id = d[i][0]
    print('Searching for CDR Product: ' + product_id)
    print('Connecting to http//moon.asu.edu')
    productURL = 'http://moon.asu.edu/planetview/inst/lroc/'+product_id
    print('Connected!')
    filetype = '.tif'
    save_path = "/Volumes/DATA DISK/PDS_FILES/LROC_NAC/"+product_id+filetype

    #http://python-guide-pt-br.readthedocs.io/en/latest/scenarios/scrape/
    page = requests.get(productURL)
    tree = html.fromstring(page.content)

    downloadURL = tree.xpath('//*[@id="browseformats"]/a[4]')
    print('Identified image download URL: '+downloadURL[0].attrib['href'])
    print('Downloading...')
    #resource = urllib.urlopen(downloadURL[0].attrib['href'], 'wb')
    #time.sleep(10)
    #output = open(save_path, "wb")
    #output.write(resource.read())
    #output.close()

    saveImage(downloadURL[0].attrib['href'], save_path)

    #t = threading.Thread(target=saveImage, args=(downloadURL[0].attrib['href'], save_path))
    #t.start()
    #threads.append(t)
    #map(lambda t: t.join(), threads)

    print("Image downloaded: "+save_path)