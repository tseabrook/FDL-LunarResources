import urllib
import csv
from lxml import html
import requests


#https://stackoverflow.com/questions/16989647/importing-large-tab-delimited-txt-file-into-python
filename = 'NACstamps.txt'
file_object  = open(filename, 'r')

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
    filetype = '.tif'
    save_path = "imgs/"+product_id+filetype

    #http://python-guide-pt-br.readthedocs.io/en/latest/scenarios/scrape/
    page = requests.get(productURL)
    tree = html.fromstring(page.content)
    downloadURL = tree.xpath('//*[@id="browseformats"]/a[4]')
    print('Identified image download URL: '+downloadURL)
    print('downloading...')
    urllib.urlretrieve(downloadURL[0].attrib['href'], save_path)
    print("Image downloaded: "+save_path)