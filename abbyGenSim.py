# import modules & set up logging
import pickle as pickle
import string
import gensim
import logging


vocabFname = '/Users/mcmenamin/GitHub/word2vec_advice/wordDict.npy'
mmapVecsFname = '/Users/mcmenamin/GitHub/word2vec_advice/wordVecs.npy'

with open(vocabFname, 'rb') as file:
    vocabDict = pickle.load(file)

idxToVocab = {vocabDict[i]: i for i in vocabDict}

wordVecs = np.load(mmapVecsFname, mmap_mode='r')


###########################################
###########################################
##
## Import text data that had been scraped
##

"""
We start out by imprting the text data we'd scraped [sp?], and then
tokenize each Q's and A's into the multiword phrases in our google
vocabularly. Full phrases that are also in the stopword list are dropped
"""

df_text = pd.read_pickle('../scrape_scripts/abbyText.pickle')


doAllTokenizing = False

def tokenizeDocstr(docStr, vocab, stopWords=[]):
    stopWords += ['_', '']
    docStr = docStr.strip().translate(transTable).lower().split()
    wordList = []
    for phrLen in range(4, 0, -1):
        for i in range(0, len(docStr) - phrLen):
            tmpPhrase = '_'.join(docStr[i:(i + phrLen)])
            if tmpPhrase in vocab:
                wordList.append(tmpPhrase)
                docStr[i:(i + phrLen)] = [''] * phrLen

    wordList = [w for w in wordList if w not in stop]
    return wordList


if doAllTokenizing:

    # get stop words
    SW = set()
    for line in open('/Users/mcmenamin/GitHub/word2vec_advice/stop_words.txt'):
        line = line.strip()
        if line != '':
            SW.add(line)
    stop = list(SW)
    transTable = {ord(i): None for i in string.punctuation}

    allSentences = []
    for idx, a in enumerate(allQs):
        if (idx % 100) == 0:
            print('{}, {:.0f}%'.format(idx, (100 * idx) / len(allQs)))
        for i in range(len(a.Q)):
            tmp = gensim.models.doc2vec.TaggedDocument(tokenizeDocstr(a.Q[i], vocabDict, stopWords=stop), ["{}_Q{}".format(idx, i)])
            allSentences.append(tmp)
        for i in range(len(a.A)):
            tmp = gensim.models.doc2vec.TaggedDocument(tokenizeDocstr(a.A[i], vocabDict, stopWords=stop), ["{}_A{}".format(idx, i)])
            allSentences.append(tmp)

    with open('/Users/mcmenamin/GitHub/word2vec_advice/tokenAbby.pickle', 'wb') as file:
        pickle.dump(allSentences, file)


with open('/Users/mcmenamin/GitHub/word2vec_advice/tokenAbby.pickle', 'rb') as file:
    allSentences = pickle.load(file)







###########################################
###########################################
##
## Test distribution-based document using mean w2v for each document
##


def getFullText(lab):
    tmp = lab.split('_')
    i1 = int(tmp[0])
    i2 = int(tmp[1][1:])
    if tmp[1][0] == 'Q':
        return allQs[i1].Q[i2]
    else:
        return allQs[i1].A[i2]


def docMeanVec(testDoc):
    idx = [vocabDict[t] for t in testDoc]
    return np.mean(wordVecs[idx, :], axis=0)

allDocMeans = np.vstack([docMeanVec(a.words) for a in allSentences])
allDocMeans[np.isnan(allDocMeans)] = 0.0
allDocMeans[np.isinf(allDocMeans)] = 0.0
allDocMeans /= np.sqrt(np.sum(allDocMeans**2, axis=1, keepdims=True))
allDocMeans[np.isnan(allDocMeans)] = 0.0
allDocMeans[np.isinf(allDocMeans)] = 0.0

targetDoc = 0
simMat_mean = allDocMeans.dot(allDocMeans[targetDoc:(targetDoc + 1), :].T)
simMat_mean[np.isnan(simMat_mean)] = -5


###########################################
###########################################
##
## Test distribution-based document using distro w2v for each document
##

"""
The mean-vector approach works really well if you have short documents
about a single concept, but doesn't make sense for longer doucments that
may be 'multi-modal' in topic space (i.e., the address several distinct)
topics. So, we can use the following metric to analyze similarity between
the distribution of topics across all concept space.

The method works as follows:
1) The content of each document is modeled as a distribution over all of
   topic space, rather than a single vector. We create the distribtion as a
   sum of gaussians, one centered on each word-vector from the document
   and having standard deviation sigma=s
 2) similarity between documents is measured as the total overlap between
    distributions for two documents (i.e., the integral of their joint prob
    over all of semantic space)
"""

from scipy.spatial.distance import pdist, cdist, squareform


def doc2vecMat(testDoc):
    idx = [vocabDict[t] for t in testDoc]
    return wordVecs[idx, :]


def btwnMag(doc1, doc2=None, sigma=1.0):
    if doc2 is not None:
        dist = cdist(doc1, doc2, metric='sqeuclidean')
    else:
        dist = squareform(pdist(doc1, metric='sqeuclidean'))
    dist = np.exp(-sigma * dist)
    mag = np.mean(dist.ravel())
    return mag


def docDist(doc1, doc2, s=1.0):
    m1 = btwnMag(doc1, sigma=s)
    m2 = btwnMag(doc2, sigma=s)
    bw = btwnMag(doc1, doc2=doc2, sigma=s)
    return bw / np.sqrt((m1 * m2))

seed = 0
seedDoc = doc2vecMat(allSentences[seed].words)
simMat_distro = [docDist(seedDoc, doc2vecMat(a.words), s=1.0) for a in allSentences]


# comparing the similarity of seed document to everything else using both metrics

from scipy.stats import rankdata

plt.close('all')
plt.subplot(121)
plt.scatter(simMat_distro, simMat_mean)
plt.ylabel('Similariity using meanVec')
plt.xlabel('Similariity using distros')

plt.subplot(122)
plt.scatter(rankdata(simMat_distro), rankdata(simMat_mean))
plt.ylabel('RANK Similariity using meanVec')
plt.xlabel('RANK Similariity using distros')


"""
This scatter plot shoes that the agree for the most part (strong
positive relationship), but there are some differences. Whether or not this is
better will probably hing upon the particular use case and average document
length/complexity.
"""


def printNearestText(dist, topN=3):
    dist = np.array(dist).ravel()
    dist[np.isnan(dist)] = -5.0
    cutoff = np.sort(dist)[-(topN + 1)]
    for i in where(dist > cutoff)[0]:
        print('**** {:.2f} ************'.format(dist[i]))
        print(getFullText(allSentences[i].tags[0]))

print('**** SEED ************')
print(getFullText(allSentences[targetDoc].tags[0]))
printNearestText(simMat_mean, topN=5)

printNearestText(simMat_distro, topN=5)
