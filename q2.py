import numpy as np
from os import listdir
import os.path
import ipdb;

fp_stopwords = open("stopwords.txt",'r')
stopwords = (fp_stopwords.read()).split('\n')

word_dict = {}
doc_count = 1
ignorechars = '''."/?[]{}(),:'!'''

for direc in listdir('./q2data/train'):
    locs = './q2data/train/'+direc+'/'
    for filename in listdir(locs):
        fp_data = open(locs+filename,'r')
        tokens = (fp_data.read()).split(' ')

        for w in tokens:
            w = w.lower().translate(None, ignorechars)
            if w in stopwords:
                continue
            if w in word_dict:
                word_dict[w].append(doc_count)
            else:
                word_dict[w] = [doc_count]
        doc_count += 1
#files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

ipdb.set_trace()
