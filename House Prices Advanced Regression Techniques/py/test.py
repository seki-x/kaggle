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
#creating matrices for sklearn:
X = all_data[:train.shape[0]]
test_data = all_data[train.shape[0]:]
y = train.SalePrice
del train

from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import make_scorer, mean_squared_error

def rmse(truth, pred):
    return np.sqrt(mean_squared_error(truth, pred))
#rmse_scorer = make_scorer(rmse, greater_is_better=False)
def rmseCvMean(model, X, y, cv=5, random_state=41):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, random_state=random_state)
    scr = 0
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scr += rmse(y_test, pred)
        print('\t', rmse(y_test, pred))
    return scr/cv

def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return rmse
def rmse_cv_mean(model, X, y):
    return rmse_cv(model, X, y).mean()
#%% import models
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR,SVR
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#%% Coefficient Vis
def coeff_imp_vis(X=X):
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
    model_lasso.fit(X, y)
    rmse_cv_mean(model_lasso, X, y).mean()
    coef = pd.Series(model_lasso.coef_,index=X.columns)
    imp_coef = pd.concat([coef.sort_values().head(10),
                         coef.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (5, 5)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()
    
    model_rf = RandomForestRegressor().fit(X, y)
    coef_rf = pd.Series(model_rf.feature_importances_,index=X.columns)
    imp_coef = pd.concat([coef_rf.sort_values().head(10),
                         coef_rf.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (5, 5)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the RF Model")
    plt.show()

    return coef, coef_rf
coef_lasso, coef_rf = coeff_imp_vis()
imp_lasso_srt = coef_lasso.sort_values(ascending=False)
imp_rf_srt = coef_rf.sort_values(ascending=False)

#%% feature eng
CV_params = [10, 1, 0.1, 0.001, 0.0005]
def select_feat(X, y, Xtest, imp_srt, model=ElasticNetCV(alphas=CV_params)):
    scr = []
    imp_feat = list(imp_srt.index[:])
    for i in range(len(imp_feat)):
        top_Ifeat = imp_feat[:i+1]
        scr.append(rmse_cv_mean(model, X[top_Ifeat], y))
        if i % 15 == 0:
            print('Top %s featrues: CV %.6f' % (i, scr[-1]))
    plt.plot(scr)
    chosen_feat = imp_feat[:scr.index(min(scr))+1]
    return X[chosen_feat], Xtest[chosen_feat]

X_slct_lasso, test_slct_lasso = select_feat(X, y, test_data, imp_lasso_srt, model=ElasticNetCV(alphas=CV_params))
X_slct_rf, test_slct_rf = select_feat(X, y, test_data, imp_rf_srt, model=ElasticNetCV(alphas=CV_params))

rmse_cv_mean(ElasticNetCV(alphas=CV_params), X_slct_lasso, y)
rmse_cv_mean(ElasticNetCV(alphas=CV_params), X_slct_rf, y)

X = X_slct_rf
test_data = test_slct_rf
#%% feature eng
imp_feat1 = list(imp_lasso_srt.index[:3])
imp_feat2 = list(imp_rf_srt.index[:3])
imp_feat = list(set(imp_feat1).union(set(imp_feat2)).
                   intersection(set(numeric_feats)))
def quad_feature(arr):
    columns_new = [colname + '_quad' for colname in arr.columns]
    arr_new = pd.DataFrame(arr ** 2)
    arr_new.columns = columns_new
    return arr_new

quad_feat = quad_feature(X[imp_feat])
X_quad = pd.concat((X, quad_feat), axis=1)
CV_params = [10, 1, 0.1, 0.001, 0.0005]
model_ela = ElasticNetCV(alphas=CV_params) #LassoCV(alphas=CV_params)
rmse_no_quad = rmse_cv_mean(model_ela, X, y)
rmse_quad = rmse_cv_mean(model_ela, X_quad, y)
_,_ = coeff_imp_vis(X_quad)

quad_feat = quad_feature(test_data[imp_feat])
test_data_quad = pd.concat((test_data, quad_feat), axis=1)

#%% single model
#X = X_quad.values
#test_X = test_data_quad.values
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
#%% residuals
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
for name, model in models.items():
    residuals = pd.DataFrame({name:train_preds[name], "true":y})
    residuals["residuals"] = residuals["true"] - residuals[name]
    residuals.plot(x = name, y = "residuals",kind = "scatter")

#%% Error-adaptive

class Boosting(object):
    def __init__(self,_base_model,_base_params,_mentor_model,_mentor_params,_maxIter=50):
        self.mentor_model = _mentor_model(**_mentor_params)
        self.model_li = [_base_model(**_base_params) for i in range(_maxIter)]
        self.model_wei = [1/_maxIter for i in range(_maxIter)]
        self.model_maxIter = _maxIter
        self.pred_li = None
        self.scores = []
    def fit(self, X, y):
        predZ = np.zeros(np.shape(X[:, 0]))
        Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
        predZ =  self.mentor_model.fit(Xz, y).predict(Xz)
        Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
#        plt.plot(sorted(predZ))
        scr = []
        for model in self.model_li:
            predZ = model.fit(Xz, y).predict(Xz)
            Xz = np.concatenate((X, np.reshape(predZ,(np.size(predZ),1))),axis=1)
            scr.append(rmse(y, predZ))
            print(scr[-1])
#        plt.plot(scr)
#        plt.show()
#            plt.plot(sorted(predZ))
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

params = {'alphas': CV_params,
#          'l1_ratio': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
#          'max_iter': 1e4,
          'cv': 5,}
bst = Boosting(ElasticNetCV,params,
               ElasticNetCV,params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)
bst.fit(X_train, y_train)
print(rmse(y_test, bst.predict(X_test)))

print(rmse(y, bst.predict(X)))
print(rmse(y, ElasticNetCV(alphas=CV_params).fit(X_train, y_train).predict(X)))

#print(rmseCvMean(bst, X, y))
#print(rmseCvMean(ElasticNetCV(alphas=CV_params), X, y))

model_ela = ElasticNetCV(alphas=CV_params).fit(X_train, y_train)
model_lasso = ElasticNetCV(alphas=CV_params).fit(X_train, y_train)
#%% submission
name = 'error_ada_tune_featSlct'

pred = bst.predict(test_X)
solution = pd.DataFrame({"id":test.Id, "SalePrice":np.expm1(pred)})
solution.to_csv(r'./submissions/' + name + ".csv", index = False)






