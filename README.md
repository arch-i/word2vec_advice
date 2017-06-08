# word2vec_advice

using word2vec to find advice in Dear Abby columns

## `./scrape_scripts`

Contains a script that scrapes "Dear Abby" advice columns, parses the text into Question/Answer pairs.
The dataset is stored as a python3 pickled pandas dataframe (`abbyText.pickle`) and an ASCII json-formated plaintext (`abbyText.json`)


## `./embeddings`

Contains a few data files that contain pre-trained word2vec embeddings derived from the Google News embeddings (`GoogleNews-vectors-negative300.bin.gz`). Specifically, there are two numpy files:
* A dictionary that maps a token (string) to an index (int)
* A memory-mapped matrix of size (embedding dimension)-by-(vocab size).

Using these two matrices, you can quickly look up the embedding vector for a particular word from the memory mapped embedding matrix without a ton of RAM overhead or long load times. lo