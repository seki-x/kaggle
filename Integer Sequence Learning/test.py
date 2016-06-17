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
sample = train['Sequence'][21] #id = 3
sep = sample.split(',')
tmp = []
for x in sep:
    if x[0]=='-':
        

length = [len(x) for x in sep]
pos = [str(x) for x in length]
feature = pd.DataFrame({'length':length})
