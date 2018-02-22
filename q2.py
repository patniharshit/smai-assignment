import numpy as np
from os import listdir
import os.path
from nltk.stem import PorterStemmer
import ipdb;

ps = PorterStemmer()

fp_stopwords = open("stopwords.txt",'r')
stopwords = (fp_stopwords.read()).split('\n')

word_dict = {}
doc_count = 1
ignorechars = {'''."/?[]{}(),:'!''' : None}
temp = 0
for direc in listdir('./q2data/train'):
    locs = './q2data/train/'+direc+'/'
    for filename in listdir(locs):
        fp_data = open(locs+filename,'r')
        try:
            tokens = ((fp_data.read()).decode("utf8")).split(' ')
        except:
            temp += 1
            continue
        for w in tokens:
            w = ps.stem((w.lower()).translate(ignorechars))
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
DocsPerWord = np.sum(np.asarray(A > 0, 'i'), axis=1)
rows, cols = A.shape
for i in range(rows):
    for j in range(cols):
        A[i,j] = (A[i,j] / WordsPerDoc[j]) * np.log(float(cols) / DocsPerWord[i])


ipdb.set_trace()
