# -*- coding: utf-8 -*-
'''
Created on Wed Jun 15 18:26:43 2016

@author: john
'''
import pandas as pd
import numpy as np

#%% Loading data
def loadData(fileName):
    '''
    Read data from file
    '''
    data = pd.read_csv(fileName)
    return data

#fileName = './data/*.csv'
train = loadData('./data/train.csv')
test = loadData('./data/test.csv')

#%% Exploration
sample = train['Sequence'][21]
sep = sample.split(',')
def popMinus(series):
    tmp = []
    ifMinus = []
    for x in series:
        if x[0] == '-':
            ifMinus.append(1)
            tmp.append(x[1:])
        else:
            ifMinus.append(0)
            tmp.append(x)
    return tmp,ifMinus

sep, ifMinus = popMinus(sep)
length = [len(x) for x in sep]
pos = [str(x) for x in length]
idx = range(len(sep))


feature = pd.DataFrame({'length':length,'#idx':idx,'ifMinus':ifMinus,})
