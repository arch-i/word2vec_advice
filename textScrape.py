#http://docs.python-guide.org/en/latest/scenarios/scrape/ 
#http://frankbi.io/AAJA-Scraper/tutorial.html
from lxml import html
import requests

import re as re
import numpy as np
import pickle as pickle

def scrape_page(extURL):    
    page = requests.get('{0}{1}'.format(baseURL,extURL))
    tree = html.fromstring(page.text)
    questions = tree.find_class('item-section')
    allQ = []
    for q in questions:
        qText = [i.text_content() for i in q.iterfind('p')]
        allQ.append(qText)
    return allQ


def parseAbby(block):
    validQA = False
    Q = []
    A = []
    dearBlock = [re.match('DEAR [a-zA-Z ]+[:,\- ]+',p) for p in block]
    abbyBlock = [re.match('DEAR ABBY[:,\- ]+',p) for p in block]
    signoffBlock = [re.search(' [\-.]+ [A-Z,. ]+',p) for p in block]
    
    # Which paragraphs are the starts of questions, starts of answers
    Qstart = [i for i in range(len(block)) if (abbyBlock[i]!=None)]
    Astart = [i for i in range(len(block)) if (dearBlock[i]!=None) & (abbyBlock[i]==None)]
    validQA = (len(Qstart)>0)&(len(Astart)>0)
    
    if validQA:
        # Anything that begins 'dear abby' and isn't followed by an answer is
        # part of the answer as well -- usually another reader providing additional
        # advice
        Qstart = Qstart[:len(Astart)]
        numQApairs = len(Qstart)
        
        #Trim out salutation/signoffs
        for i in range(len(block)):
            onset = 0
            offset = len(block[i])
            if dearBlock[i]!=None:
                onset = np.max((onset,dearBlock[i].span()[1]))
            if abbyBlock[i]!=None:
                onset = np.max((onset,abbyBlock[i].span()[1]))
            if signoffBlock[i]!=None:
                offset = np.min((offset,signoffBlock[i].span()[0]))
            block[i] = block[i][onset:offset]
        
        for i in range(numQApairs):
            tmpQ = ' '.join(block[Qstart[i]:Astart[i]])
            if i<(numQApairs-1):
                tmpA = ' '.join(block[Astart[i]:Qstart[i+1]])
            else:
                tmpA = ' '.join(block[Astart[i]:])
            if len(tmpQ)>0:
                Q.append(tmpQ)
            if len(tmpA)>0:
                A.append(tmpA)
    validQA = (len(Q)>0)&(len(A)>0)

    return Q,A,validQA

class webtext(object):
    def __init__(self,url,rawText,bNum):
        self.validQA = False
        self.url = url
        self.blockNumber = bNum
        self.rawText = rawText
        self.Q = []
        self.A = []
        self.Qvec = []
        self.Avec = []
    
    def parseQA(self,parseMethod):
        self.Q,self.A,self.validQA = parseMethod(self.rawText)
    
    def makeVecQA(self):
        self.Qvec = [toVec(q) for q in self.Q]
        self.Avec = [toVec(a) for a in self.A]
    


baseURL = 'http://www.uexpress.com/'
archiveURL = 'http://www.uexpress.com/dearabby/archives'

archiveList = []
for year in range(1991,2015+1):
    archive = requests.get('{0}/{1}'.format(archiveURL,year))
    tree = html.fromstring(archive.text)
    yearArchive = [a.attrib['href'] for a in tree.find_class('media-link-main')]
    archiveList += yearArchive

allQs = []
for url in archiveList:
    print(url)
    articleText = scrape_page(url)
    for i,block in enumerate(articleText):
        allQs.append(webtext(url,block,i))
        allQs[-1].parseQA(parseAbby)

save_file = '/home/bmc/Dropbox/codeProjects/allAbby.pickle'
with open(save_file,'wb') as file:
    pickle.dump(allQs,file)


#
#with open('/home/bmc/Dropbox/codeProjects/allAbby.txt','w') as file:
#    for idx,a in enumerate(allQs):
#        if a.validQA:
#            for i in range(len(a.Q)):
#                file.write("{}_Q\t{}\n".format(idx,a.Q[i]))
#            for i in range(len(a.A)):
#                file.write("{}_A\t{}\n".format(idx,a.A[i]))
#    file.close()