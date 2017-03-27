# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 19:05:39 2017

@author: xjz19
"""

import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import scipy
import codecs
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from nltk.stem.porter import PorterStemmer

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import \
RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,\
GradientBoostingClassifier,VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression,\
 RidgeClassifier, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.decomposition import TruncatedSVD#,PCA not supported for sparse-mat

#%% read csv files
def load_data(fdir):
#    fdir = r'.\\..\\input\\'
    fname_li = os.listdir(fdir)
    data = dict()
    for fname in fname_li:
        data[fname.split('.csv')[0]] = pd.read_csv(fdir+fname)
    return data

data = load_data(r'.\\..\\input\\')
sample_submission = data['sample_submission']
test_data = data['test']
del data['sample_submission'], data['test']

data_merge = pd.DataFrame(columns=['title','content','tags','category'])
for category, df in data.items():
    del df['id']
    df['category'] = category
    data_merge = pd.concat((data_merge,df),axis=0)

tags = MultiLabelBinarizer().fit_transform([i.split(' ') for i in data_merge['tags']])
categories = MultiLabelBinarizer().fit_transform([i.split(' ') for i in data_merge['category']])
posts_raw = data_merge.drop(['tags','category'],axis=1)
del data_merge,data,df

#%% cleaning
def clean_text(df):
    stemmer = PorterStemmer()
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return ' '.join(stems)
    intab = string.punctuation
    outtab = "                                "
    trantab = str.maketrans(intab, outtab)
    for colname in df.columns:
        #--- each col
        corpus = []
        for text in df[colname]:
            soup = BeautifulSoup(text,"lxml")
            text = soup.get_text()
            text = text.lower()
            text = text.translate(trantab)
            text = tokenize(text)
            corpus.append(text)
        df[colname] = corpus
    return df
#posts_clean = clean_text(posts_raw)
#posts_clean.to_csv('posts_clean.csv',encoding='utf-8',index=False, header=True)
posts = pd.read_csv('posts_clean.csv')
del posts_raw

title_tfidf = TfidfTransformer().fit_transform(
        CountVectorizer().fit_transform(posts['title']))
content_tfidf = TfidfTransformer().fit_transform(
        CountVectorizer().fit_transform(posts['content']))
#X = np.concatenate((title_tfidf.toarray(),content_tfidf.toarray()),axis=1) # memory error!!!

#%% dimention reduction
def TSVD(X):
    start_time = time.time()
    data_svd = TruncatedSVD(np.shape(X)[1]-1).fit(X)
#    data_svd = TruncatedSVD(4000).fit(X)
    fit_time = time.time() - start_time
    print('TruncatedSVD'+' is over in %.2f sec...'%(fit_time))
    accumulate_ratio = [sum(data_svd.explained_variance_ratio_[:i])\
                            for i in range(np.shape(X)[1])]
    dr_thrs = .8
    for i in accumulate_ratio:
        if i > dr_thrs:
            n_select = accumulate_ratio.index(i)
            break
    plt.plot(accumulate_ratio)
    plt.axhline(y=dr_thrs, color='g')
    plt.axvline(x=n_select,linewidth=4,color='k')
    plt.axis([0, n_select*2, 0, 1])
    plt.show()
    plt.close()
    X_select = TruncatedSVD(n_select).fit_transform(X)
    X_select = scipy.sparse.csr_matrix(X_select)
    return X_select

title_dr = TSVD(title_tfidf)

content_dr = TSVD(content_tfidf)

X = np.concatenate((title_dr.toarray(),content_dr.toarray()),axis=1)
X = scipy.sparse.csr_matrix(X)
y = categories

