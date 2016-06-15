# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:26:43 2016

@author: john
"""
import scipy as sp
import pandas as pd
import numpy as np

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def loadData(fileName):
    '''
    Read data from file
    '''
    data = pd.read_csv(fileName)
    return data

#fileName = './data/data.csv'
data = loadData('./data/data.csv')
data.shot_made_flag = data.shot_made_flag.fillna(2)
train = data[data["shot_made_flag"]!=2].drop(["game_event_id","game_id","lat","lon","team_name","shot_id"],axis=1)
test = data[data["shot_made_flag"]==2].drop(["game_event_id","game_id","lat","lon","team_name","shot_id"],axis=1)