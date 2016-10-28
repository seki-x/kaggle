# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:44:34 2016

@author: john
"""

import pandas as pd
import numpy as np
import seaborn as sns
#import matplotlib
import matplotlib.pyplot as plt

#%% Load data 

train = pd.read_csv(r'..\input\train.csv')
test = pd.read_csv(r'..\input\test.csv')

all_data = pd.concat((train.loc[:,'cat1':'cont14'],\
                     test.loc[:,'cat1':'cont14']))
test_id = test.id

#%% Preprocessing

#y_loss_1_log1p = pd.DataFrame({"loss":train["loss"], \
#                       "log(loss + 1)":np.log1p(train["loss"])})
#y_loss_1_log1p.hist()

#from scipy.stats import skew
#numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index
#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
#y = np.log1p(train["loss"])
y = train["loss"]
#%% Models

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

model_rf = RandomForestRegressor(
#                                 n_estimators=50,\
                                 criterion='mae',\
                                 n_jobs=-1)

sub_ratio = 0.08
sub_num = np.floor(sub_ratio * np.size(X_train,0))
sub_X_train = X_train.loc[:sub_num,:]
sub_y = y.loc[:sub_num]

#rf = model_rf.fit(sub_X_train, sub_y)

def mae_cv(model, X = sub_X_train, y = sub_y):
    mae = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv = 5)
    return(mae)

rf_mae_cv = mae_cv(model_rf)


#solution = pd.DataFrame({"id":test_id, "loss":pred})
#solution.to_csv("sol.csv", index = False)














