# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:49:35 2017

@author: xjz19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
import scipy

from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics 
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
#%% read csv files
def load_data(fdir):
#    fdir = r'.\\..\\input\\'
    fname_li = os.listdir(fdir)
    data = dict()
    for fname in fname_li:
        if '_light' in fname:
            category = fname.split('_light')[0]
            data[category] = pd.read_csv(fdir+fname)
            del data[category]['id']
    return data

data = load_data(r'.\\..\\input\\')
test_data = data['test']
del data['test']
category = list(data.keys())

example = data['diy']
mlb = MultiLabelBinarizer()
multi_labels = mlb.fit_transform([i.split(' ') for i in example['tags']])
labels_name = list(mlb.classes_)

title = example['title']
content = example['content']
#del example

t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(title)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
y_true = multi_labels
k_true = np.shape(y_true)[1]

km = MiniBatchKMeans(n_clusters=k_true, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000)
print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
cls = []
for i in range(k_true):
    clsTerms = []
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind], end='')
        clsTerms.append(terms[ind])
    cls.append(clsTerms)
    print()
del clsTerms
clusters = [i[0:3] for i in cls]

mlb = MultiLabelBinarizer()
multi_clusters = mlb.fit_transform(clusters)
clusters_name = list(mlb.classes_)

multi_total = mlb.fit_transform((example['tags'],clusters))
