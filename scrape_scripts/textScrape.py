#!/Users/mcmenamin/.virtualenvs/py2env/bin/python

from lxml import html
import requests

from datetime import date
import numpy as np
import pandas as pd

import re as re

from itertools import chain
import cPickle as pickle

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
    return allQ

def parseAbby(block):
    dearBlock = [re.match('DEAR [a-zA-Z. ]+[:,\- ]+', p) is not None for p in block]
    abbyBlock = [re.match('DEAR ABBY[:,\- ]+', p) is not None        for p in block]
    
    # Which paragraphs are the starts of questions, starts of answers
    Qstart = [i for i in range(len(block)) if abbyBlock[i]]
    Astart = [i for i in range(len(block)) if dearBlock[i] and not abbyBlock[i]]
    
    QA_pairs = [[]]
    if len(Qstart)>0 and len(Astart)>0:
        numQApairs = len(Qstart)

        Q = [' '.join(block[i[0]:i[1]]) for i in zip(Qstart, Astart)]
        A = [' '.join(block[i[0]:i[1]]) for i in zip(Astart, Qstart[1:]+[len(block)])]
        QA_pairs = zip(Q, A)
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