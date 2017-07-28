import urllib2
import csv
from lxml import html
import requests
import threading


def saveImage(url, path):
    req = urllib2.Request(url)
    resp = urllib2.urlopen(req)
    imgdata = resp.read()
    with open(path, 'wb') as outfile:
        outfile.write(imgdata)

#https://stackoverflow.com/questions/16989647/importing-large-tab-delimited-txt-file-into-python
filename = 'NACstamps.txt'
file_object  = open(filename, 'r')

threads = []
d = []
with open(filename,'rb') as source:
    for line in source:
        fields = line.split('\t')
        d.append(fields)

for i in range(1,len(d)):
    product_id = d[i][0]
    print('Searching for CDR Product: ' + product_id)
    print('Connecting to http//moon.asu.edu')
    productURL = 'http://moon.asu.edu/planetview/inst/lroc/'+product_id
    print('Connected!')
    filetype = '.tif'
    save_path = "imgs/"+product_id+filetype

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