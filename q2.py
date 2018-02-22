import numpy as np
from os import listdir
import os.path
from nltk.stem import PorterStemmer
import math

ps = PorterStemmer()

fp_stopwords = open("stopwords.txt",'r')
stopwords = (fp_stopwords.read()).split('\n')

word_dict = {}
doc_count = 0
ignorechars = {'''."/?[]{}(),:'!''' : None}

for direc in listdir('./q2data/train'):
    locs = './q2data/train/'+direc+'/'
    files = listdir(locs)
    for filename in files[:(80*len(files))/100]:
        fp_data = open(locs+filename,'r')
        tokens = ((fp_data.read()).decode("utf8")).split(' ')
        for w in tokens:
            w = (w.lower()).translate(ignorechars)
            w = ps.stem(w)
            if w in stopwords:
                continue
            if w in word_dict:
                word_dict[w].append(doc_count)
            else:
                word_dict[w] = [doc_count]
        doc_count += 1

dictkeys = word_dict.keys()
#dictkeys.sort()
A = np.zeros([len(dictkeys), doc_count])

for i, k in enumerate(dictkeys):
    for d in word_dict[k]:
            A[i,d] += 1
#files = [f for f in listdir(mypath) if isfile(join(mypath, f))]


WordsPerDoc = np.sum(A, axis=0)
DocsPerWord = np.sum(np.asarray(A > 0), axis=1)
rows, cols = A.shape

for i in range(rows):
    for j in range(cols):
            A[i,j] = (A[i,j] / WordsPerDoc[j]) * math.log(1 + (float(cols) / DocsPerWord[i]))
