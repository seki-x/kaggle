# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:00:36 2017

@author: John
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
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from nltk.stem.porter import PorterStemmer

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


tags = data['robotics']['tags']
multi_labels = MultiLabelBinarizer().fit_transform([i.split(' ') for i in tags])

posts = data['robotics'].drop(['id','tags'],axis=1)
#%% cleaning
def clean_text(df):
    stemmer = PorterStemmer()
    #from nltk.corpus import stopwords
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        #tokens = [word for word in tokens if word not in stopwords.words('english')]
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
    
posts_clean = clean_text(posts)       
title_tfidf = TfidfTransformer().fit_transform(
        CountVectorizer().fit_transform(posts_clean['title']))
content_tfidf = TfidfTransformer().fit_transform(
        CountVectorizer().fit_transform(posts_clean['content']))

X = np.concatenate((title_tfidf.toarray(),content_tfidf.toarray()),axis=1)
X = scipy.sparse.csr_matrix(X)
y = multi_labels

#%% classifiers
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

#%% dimention reduction
start_time = time.time()
data_svd = TruncatedSVD(np.shape(X)[1]-1).fit(X)
fit_time = time.time() - start_time
print('TruncatedSVD'+' is over in %.2f sec...'%(fit_time))
accumulate_ratio = [sum(data_svd.explained_variance_ratio_[:i])\
                        for i in range(np.shape(X)[1])]
dr_thrs = .95
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

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=.3, random_state=42)
#%% Single model
predictions = dict()
scores = dict()
time_costs = dict()

#==============================================================================
#    "RidgeClassifier": OneVsRestClassifier(RidgeClassifier()), #.31.31 SVDslow
#    "Neural Net": MLPClassifier(), #.28.25 slow
#    "DecisionTreeClassifier": DecisionTreeClassifier(), #.28.14 SVDslow
#    "Random Forest": RandomForestClassifier(), #.26.03
#    "ExtraTrees": ExtraTreesClassifier(), #.26    
#    "Logistic Regression": OneVsRestClassifier(LogisticRegression()), #.10 slow
#    "Linear Discriminant Analysis": OneVsRestClassifier(LinearDiscriminantAnalysis()), #.27 slow
#    "KNN": KNeighborsClassifier(), #.23
#    "Linear SVM": OneVsRestClassifier(LinearSVC()), #.34.34
#    "AdaBoost": OneVsRestClassifier(AdaBoostClassifier()), #.33
#   BernoulliNB, MultinomialNB, GaussianNB #.00~.16
#==============================================================================

classifiers = {
#    "GradientBoosting": OneVsRestClassifier(GradientBoostingClassifier()), #.35
    "PassiveAggressiveClassifier": OneVsRestClassifier(PassiveAggressiveClassifier()), #.36.34
    "SGD": OneVsRestClassifier(SGDClassifier()), #.36.34

#    "DecisionTreeBagging": OneVsRestClassifier(BaggingClassifier()), #.37
#    "Voting": OneVsRestClassifier(
#            VotingClassifier([
#                    ('SGD',SGDClassifier(loss='log')),
#                    ('Bagging',BaggingClassifier()),
#                    ('GradientBoosting',
#                     GradientBoostingClassifier())],
#                     voting='soft')),#.38
    }

cnt = 0
for name, clf in classifiers.items():
    cnt += 1
    print('\nTraining '+name+' No.%d CLF in all...'%(cnt))
    start_time = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(name+' is trained in %.2f sec No.%d in all...'%(fit_time,cnt))
    print('Prediction of '+name+' No.%d CLF in all...'%(cnt))
    start_time = time.time()
    try:
        pred = clf.predict(X_test)
    except TypeError:
        pred = clf.predict(X_test.toarray())
    predict_time = time.time() - start_time
    predictions[name] = pred
    print(name+' predicted in %.2f sec No.%d in all...'%(predict_time,cnt))
    scores[name] = f1_score(y_test,pred,average='micro')
    print(name+' has a score of %.2f No.%d in all...'%(scores[name],cnt))
    time_costs[name] = [fit_time, predict_time]
    print('Over training %s No.%d in all...'%(name,cnt))