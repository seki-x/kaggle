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
            ifMinus.append(-1)
            tmp.append(x)
    return tmp,ifMinus

sep, ifMinus = popMinus(sep)
length = [len(x) for x in sep]
pos = [str(x) for x in length]
idx = range(len(sep))

feature1 = pd.DataFrame({'length':length,'#idx':idx,'ifMinus':ifMinus,})

train_num = 12
x = np.array(range(train_num));x = x[:,None]
y1 = feature1.values[:train_num,1]
y2 = feature1.values[:train_num,2]
yt1 = feature1.values[train_num:,1]
yt2 = feature1.values[train_num:,2]
xt = np.array(range(train_num,len(length)));xt = xt[:,None]
from sklearn.linear_model  import LinearRegression,LogisticRegression
reg1 = LogisticRegression()
reg2 = LinearRegression()
reg1 = reg1.fit(x,y1)
reg2 = reg2.fit(x,y2)
output1 = reg1.predict(xt)
output2 = reg2.predict(xt).astype(int)


















