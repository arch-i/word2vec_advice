#!/Users/mcmenamin/.virtualenvs/py3env/bin/python


"""
This script takes the big google-provided word2vec vectors
and does the following processing steps:

- clean up the vocab by forcing everything to lowercase,
  trimming out punctuation, etc. If there are multiple vectors
  that map for the same cleaned word (e.g., Dog and dog), those
  vectors are averaged

- save a dictionary that maps each word to a vector index

- save an array of the normalized vector embeddings that can
  be loaded as a memory map

"""

import numpy as np

import string
import pickle as pickle
import gensim

from tqdm import tqdm

vocabFname = './wordDict.npy'
mmapVecsFname = './wordVecs.npy'
googleWord2Vec = './GoogleNews-vectors-negative300.bin.gz'

transTable_goog = {ord(i):None for i in string.punctuation if i not in ['_']}

def cleanupGoogleword(s):
    s = s.lower().translate(transTable_goog).strip(string.punctuation)
    return s


#
# Load google's word2vec model
#

model = gensim.models.KeyedVectors.load_word2vec_format(googleWord2Vec, binary=True)


#
# Make dict of full vocab->index while keeping track of 
# which cleaned up words now map onto multiple embeddings
#

fullVocab = {val:key for key, val in enumerate(model.index2word)}
mergedVocabIdx = {}

for s in tqdm(fullVocab):
    sLower = cleanupGoogleword(s)
    if sLower in mergedVocabIdx:
        mergedVocabIdx[sLower].append(fullVocab[s])
    else:
        mergedVocabIdx[sLower] = [fullVocab[s]]


#
# Get the embeddings for each word
#

embedVecs = np.zeros((len(mergedVocabIdx), model.syn0.shape[1]))
for key, val in tqdm(enumerate(mergedVocabIdx)):
    embedVecs[key, :] = np.mean(model.syn0[mergedVocabIdx[val], :], axis=0)
embedVecs /= np.sqrt(np.sum(embedVecs**2, axis=1, keepdims=True))


#
# Save output into two files
#

with open(vocabFname, 'wb') as file:
    pickle.dump(lowerVocab, file)

np.save(mmapVecsFname, embedVecs)