#!/Users/mcmenamin/.virtualenvs/py3env/bin/python

from lxml import html
import requests

from datetime import date
import numpy as np
import pandas as pd

import re as re

from itertools import chain
import pickle

from tqdm import tqdm

def getURLforYear(year, archiveURL='http://www.uexpress.com/dearabby/archives'):
    archive = requests.get('{0}/{1}'.format(archiveURL, year))
    tree = html.fromstring(archive.text)
    urlList = [a.attrib['href'] for a in tree.find_class('media-link-main')]
    return urlList

def scrape_page(extURL, baseURL='http://www.uexpress.com/'):    
    page = requests.get('{0}{1}'.format(baseURL, extURL))
    tree = html.fromstring(page.text)
    questions = tree.find_class('item-section')
    allQ = []
    for q in questions:
        qText = [i.text_content() for i in q.iterfind('p')]
        allQ += qText
    allQ = ' '.join(allQ)
    return allQ

def parseAbby(block):
    block = block.strip().split('DEAR ')

    abbyBlock = [p.startswith('ABBY:') for p in block]
    dearReaderBlock = [p.startswith('READERS:') for p in block]
    replyBlock = [not (p[0] or p[1]) for p in zip(abbyBlock, dearReaderBlock)]
    
    QA_pairs = []
    if True in abbyBlock and True in replyBlock:
        firstBlock = abbyBlock.index(True)
        
        block = block[firstBlock:]
        abbyBlock = abbyBlock[firstBlock:]
        dearReaderBlock = dearReaderBlock[firstBlock:]
        replyBlock = replyBlock[firstBlock:]
        
        for i in range(len(block)-1):
            if abbyBlock[i] and replyBlock[i+1]:
                QA_pairs.append([block[i], block[i+1]])
    return QA_pairs


#
# Get an iterator of URLs from archives for a specific date range
#

archivedURLs = list(chain.from_iterable([getURLforYear(y) for y in range(1991,2017+1)]))


#
# Pull in the text from each archived URL
#

all_text_dict = {}
for url in tqdm(archivedURLs):
    raw_text = scrape_page(url)
    all_text_dict[url] = {'path': url,
                          'date': date(*[int(i) for i in url.split('/')[2:5]]),
                          'raw_text': raw_text,
                          'parse_text': parseAbby(raw_text)
                          }                
df_text = pd.DataFrame.from_dict(all_text_dict, orient='index')
df_text.to_pickle('abbyText.pickle')