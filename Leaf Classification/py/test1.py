# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:23:24 2016

@author: john
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Swiss army knife function to organize the data

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)
print(train.head(1))

# Stratified Train/Test Split
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Sklearn Classifier Showdown
from sklearn.metrics import accuracy_score, log_loss
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
#    KNeighborsClassifier(3),
#    SVC(kernel="rbf", C=0.025, probability=True),
#    NuSVC(probability=True),
#    DecisionTreeClassifier(),
#    RandomForestClassifier(),
#    AdaBoostClassifier(),
#    GradientBoostingClassifier(),
#    GaussianNB(),
    LinearDiscriminantAnalysis(),
#    QuadraticDiscriminantAnalysis()
    ]

# Logging for Visual Comparison

clf = classifiers[0]
clf.fit(X_train, y_train)
train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)
train_predictions = clf.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)













