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

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], 
                       "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()

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
from sklearn.svm import SVR
from sklearn.cross_validation import KFold

import datetime
#C_range = [1e4]
C_range_1 = np.logspace(0, 5, 11)
gamma_range_1 = np.logspace(-8, 3, 19)
#gamma_range =  [5.8e-7]
def gridSearch(gamma_range, C_range):
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = KFold(len(y.values), n_folds=5, shuffle=True, random_state=42)
    grid = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv=cv, scoring='mean_squared_error')
    
    starttime = datetime.datetime.now()
    grid.fit(X_train, y)
    endtime_train = datetime.datetime.now()
    timeCost_train = (endtime_train - starttime).seconds
    print('\ntraining time costed: '+str(timeCost_train)+' seconds\n')
    scores = [(-x[1])**0.5 for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    return grid, scores

#%% Grid Vis
def gridVis(grid, scores, C_range, gamma_range):
    print('\nBest cv sccore: '+str(scores.min())+'\n')
    plt.figure(figsize=(11, 9))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation')
    plt.show()

grid_1, scores_1 = gridSearch(gamma_range_1, C_range_1)
gridVis(grid_1, scores_1, C_range_1, gamma_range_1)

C_range_2 = np.linspace(C_range_1[7], 
                        C_range_1[10], 
                        9)
gamma_range_2 = np.linspace(gamma_range_1[1],
                            gamma_range_1[4],
                            11)
C_range_3 = np.linspace(C_range_2[0], 
                        C_range_2[3], 
                        8)
gamma_range_3 = np.linspace(gamma_range_2[0],
                            gamma_range_2[2],
                            6)
C_range_4 = np.linspace(C_range_3[4], 
                        C_range_3[6], 
                        6)
gamma_range_4 = np.linspace(gamma_range_3[1],
                            gamma_range_3[3],
                            6)
C_range_5 = np.linspace(C_range_4[4], 
                        C_range_3[7], 
                        6)
gamma_range_5 = np.linspace(gamma_range_4[1],
                            gamma_range_4[2],
                            6)
C_range_6 = np.linspace(C_range_5[2], 
                        C_range_5[4], 
                        7)
gamma_range_6 = np.linspace(gamma_range_5[2],
                            gamma_range_5[4],
                            7)
grid_6, scores_6 = gridSearch(gamma_range_6, C_range_6)
gridVis(grid_6, scores_6, C_range_6, gamma_range_6)

best_paras_idx = np.argwhere(scores_6==scores_6.min())
best_paras = {
            'C': C_range_6[best_paras_idx[0][0]],
            'gamma': gamma_range_6[best_paras_idx[0][1]]
            }

fin_model = SVR(kernel='rbf',
                C = best_paras['C'],
                gamma = best_paras['gamma'])
fin_model.fit(X_train, y)
preds = np.expm1(fin_model.predict(X_test))
solution = pd.DataFrame({"Id":test.Id, 
                         "SalePrice":preds})
solution.to_csv("svr.csv", index = False)
