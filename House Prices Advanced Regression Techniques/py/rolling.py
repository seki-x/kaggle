# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:57:44 2017

referred: 
    Regularized Linear Models - Alexandru Papiu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
import time
import copy

#%% load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
prices = pd.DataFrame({"price":train["SalePrice"], 
                       "log(price + 1)":np.log1p(train["SalePrice"])})
matplotlib.rcParams['figure.figsize'] = (6, 4)
prices.hist()
plt.show()

#%% preprocess
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
#creating matrices for sklearn:
X = all_data[:train.shape[0]]
test_data = all_data[train.shape[0]:]
y = train.SalePrice
trainX_raw = all_data[:train.shape[0]]
trainy_raw = train.SalePrice
del train

#%% Evaluation
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import make_scorer, mean_squared_error

def rmse(truth, pred):
    return np.sqrt(mean_squared_error(truth, pred))
#rmse_scorer = make_scorer(rmse, greater_is_better=False)
def rmseCvMean(model, X, y, cv=5, random_state=41):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, random_state=random_state)
    scr = 0
    model_li = []
    scr_li = []
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scr_li.append(rmse(y_test, pred))
        scr += rmse(y_test, pred)
        model_li.append(model)
        print('\t', rmse(y_test, pred))
    return scr/cv, model_li[scr_li.index(min(scr_li))]

def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return rmse
def rmse_cv_mean(model, X, y):
    return rmse_cv(model, X, y).mean()
#%% import models
from sklearn.linear_model import *
from sklearn.ensemble import *
#from sklearn.neural_network import MLPRegressor
#from sklearn.svm import LinearSVR,SVR
#from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.gaussian_process import GaussianProcessRegressor
#
#from sklearn.decomposition import PCA
#from sklearn.model_selection import GridSearchCV

#%% single model
X = X.values
test_X = test_data.values

train_preds = dict()
scores = dict()
test_preds = dict()
CV_params = [10, 1, 0.1, 0.001, 0.0005]

models = {
    "ElasticNetCV": ElasticNetCV(alphas=CV_params), #.1229
    "LassoCV": LassoCV(alphas=CV_params), #.1231
    }
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

cnt = 0
for name, model in models.items():
    cnt += 1
    print('\nTraining '+name+' No.%d in all...'%(cnt))
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(name+' is trained in %.4f sec No.%d in all...'%(fit_time,cnt))
    print('Prediction of '+name+' No.%d in all...'%(cnt))
    start_time = time.time()
    pred = model.predict(X_test)
    predict_time = time.time() - start_time
    train_preds[name] = model.predict(X)
    test_preds[name] = model.predict(test_X)
    print(name+' predicted in %.4f sec No.%d in all...'%(predict_time,cnt))
    scores[name] = rmse_cv_mean(model, X, y)
    print(name+' has a score of %.4f No.%d in all...'%(scores[name],cnt))
del start_time, fit_time, predict_time, pred
del cnt, name, model

#%% Rolling

class Boosting(object):
    def __init__(self,_base_model,_base_params,_mentor_model,_mentor_params,_maxIter=40):
        self.mentor_model = _mentor_model(**_mentor_params)
        self.model_li = [_base_model(**_base_params) for i in range(_maxIter)]
#        self.model_wei = [1/_maxIter for i in range(_maxIter)]
        self.model_maxIter = _maxIter
        self.pred_li = None
        self.scores = []
    def fit(self, X, y):
        predZ = np.zeros(np.shape(X[:, 0]))
        Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
        predZ =  self.mentor_model.fit(Xz, y).predict(Xz)
        Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
        scr = []
        for model in self.model_li:
            predZ = model.fit(Xz, y).predict(Xz)
            Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
            scr.append(rmse(y, predZ))
#            plt.plot(sorted(predZ))
            print(scr[-1])
        self.scores = scr
        plt.plot(scr)
#        return self

    def predict(self, X):
        predZ = np.zeros(np.shape(X[:, 0]))
        Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
        predZ =  self.mentor_model.predict(Xz)
        Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
#        plt.plot(sorted(predZ))
        for model in self.model_li:
            predZ = model.predict(Xz)
            Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
#            plt.plot(sorted(predZ))
        return predZ
CV_params = np.logspace(-5, 1, 20)
model_ela = ElasticNetCV(alphas=CV_params)
model_ela.fit(X, y)
CV_params = np.linspace(model_ela.alpha_ * 0.5, model_ela.alpha_ * 1.5, 5)
model_ela = ElasticNetCV(alphas=CV_params)
rmse_cv_mean(model_ela, X, y)

params = {'alphas': CV_params,
          'cv': 5,}
bst = Boosting(ElasticNetCV,params,
               ElasticNetCV,params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
bst.fit(X_train, y_train)

#%% cv
print(rmse(y, bst.predict(X)))
print(rmse(y, ElasticNetCV(alphas=CV_params).fit(X_train, y_train).predict(X)))

bst = Boosting(ElasticNetCV,params,
               ElasticNetCV,params, 10)

bst_cv, bst_best = rmseCvMean(bst, X, y, 5)
print(bst_cv)
ela_cv, ela_best = rmseCvMean(ElasticNetCV(alphas=CV_params), X, y, 5)
print(ela_cv)

#%% submission
name = 'rolling40_ela'
#bst.fit(X, y)
pred = bst_best.predict(test_X)
solution = pd.DataFrame({"id":test.Id, "SalePrice":np.expm1(pred)})
solution.to_csv(r'./submissions/' + name + ".csv", index = False)

