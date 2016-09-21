# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:49:22 2016

@author: john
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


prices = pd.DataFrame({"price":train["SalePrice"], 
                       "log(price + 1)":np.log1p(train["SalePrice"])})
#matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
#prices.hist()

#%% feature eng

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#%% learn
#creating matrices for sklearn
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

#%% Grid search

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score


import datetime
#C_range = [1e4]
n_estimators_range = np.linspace(1e1, 1e3, 5).astype(int)
max_depth_range = np.linspace(1e1, 1e3, 5).astype(int)
def gridSearch(n_estimators_range, max_depth_range):
    param_grid = dict(n_estimators=n_estimators_range, max_depth=max_depth_range)
    cv = KFold(len(y.values), n_folds=5, shuffle=True, random_state=42)
    grid = GridSearchCV(RandomForestRegressor(n_jobs=-1,
                                              random_state=42), 
                                              param_grid=param_grid, 
                                              cv=cv, 
                                              scoring='mean_squared_error')
    
    starttime = datetime.datetime.now()
    grid.fit(X_train, y)
    endtime_train = datetime.datetime.now()
    timeCost_train = (endtime_train - starttime).seconds
    print('\ntraining time costed: '+str(timeCost_train)+' seconds\n')
    scores = [(-x[1])**0.5 for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(n_estimators_range), 
                      len(max_depth_range))
    return grid, scores


#%% Grid Vis
def gridVis(grid, scores, n_estimators_range, max_depth_range):
    print('\nBest cv sccore: '+str(scores.min())+'\n')
    plt.figure(figsize=(11, 9))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.colorbar()
    plt.xticks(np.arange(len(max_depth_range)), max_depth_range, rotation=45)
    plt.yticks(np.arange(len(n_estimators_range)), n_estimators_range)
    plt.title('Validation')
    plt.show()

#grid_1, scores_1 = gridSearch(n_estimators_range, max_depth_range)
#gridVis(grid_1, scores_1, n_estimators_range, max_depth_range)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, 
                                   scoring="mean_squared_error", cv = 5))
    return(rmse)


alphas = np.linspace(1e1, 1e3, 5).astype(int)
cv_rf = [rmse_cv(RandomForestRegressor(n_estimators=alpha,
                                       n_jobs=-1,random_state=42)).mean() 
            for alpha in alphas]






