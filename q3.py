import numpy as np
from os import listdir
import os.path
#from nltk.stem import PorterStemmer
import math
from scipy.linalg import svd
from scipy.spatial.distance import cosine

#ps = PorterStemmer()

fp_stopwords = open("stopwords.txt",'r')
stopwords = (fp_stopwords.read()).split('\n')

word_dict = {}
doc_count = 0
ignorechars = '''."/?\[]{}(),:'!'''

for direc in listdir('./q2data/train'):
    locs = './q2data/train/'+direc+'/'
    files = listdir(locs)
    for filename in files[:(80*len(files))/100]:
        fp_data = open(locs+filename,'r')
        tokens = ((fp_data.read())).split()
        for w in tokens:
            w = w.lower().translate(None, ignorechars)
            #w = ps.stem(w)
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

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
            A[i,j] = (A[i,j] / WordsPerDoc[j]) * math.log(1 + (float(A.shape[1]) / DocsPerWord[i]))

A = A.transpose()
#w, v = np.linalg.eig(np.matmul(A, A.transpose()))
umat, smat, vh = np.linalg.svd(A, full_matrices=False)

discard = 100

smat = smat[:-discard]
umat = umat[:,:-discard]
vh = vh[:-discard,:]

locs = './q2data/train/1/'
files = listdir(locs)
for filename in files[(80*len(files))/100+1:(80*len(files))/100+2]:
    print filename
    fp_data = open(locs+filename,'r')
    tokens = ((fp_data.read())).split()
    doc_vec = np.zeros(len(dictkeys))

    word_count = 0
    for w in tokens:
        w = w.lower().translate(None, ignorechars)
        #w = ps.stem(w)
        if w in stopwords:
            continue
        elif w in dictkeys:
            word_count += 1
            doc_vec[dictkeys.index(w)] += 1

    for idx in range(len(dictkeys)):
        doc_vec[idx] = (doc_vec[idx] / word_count) * math.log(1 + (float(doc_count+1) / DocsPerWord[i]+doc_vec[idx]))

    reduced_vec = np.matmul(doc_vec, vh.transpose())

    similarity = []
    for i, uvec in enumerate(umat):
        similarity.append((1 - cosine(uvec, reduced_vec), i))

    print sorted(similarity)[-10:]
