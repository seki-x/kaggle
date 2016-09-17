# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:42:33 2016

@author: john
"""

import pandas as pd
import numpy as np

#==============================================================================
# Get sub set
#==============================================================================
#def allData(fname):
#    df = []
#    with open(fname,'r') as f:
#        df = pd.read_csv(f)
#    return df
#
#fname = r'.\data\training.csv'
#df = allData(fname)
#sub = df[:500]
#sub.to_csv(r'.\data\trainingSub.csv',index = False)
#fname = r'.\data\test.csv'
#df = allData(fname)
#sub = df[:100]
#sub.to_csv(r'.\data\testSub.csv',index = False)

def loadData(fileName):
    '''
    Read data from file
    '''
    data = pd.read_csv(fileName)
    return data

#data = loadData('./data/trainingSub.csv')

























