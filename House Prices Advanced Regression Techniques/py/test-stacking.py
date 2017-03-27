# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:30:03 2017

@author: john

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

#%% dimention reduction
def reduce_dim(X, dr_thrs = 1-1e-5):
    start_time = time.time()
    data_pca = PCA(np.shape(X)[1]).fit(X)
    fit_time = time.time() - start_time
    print('PCA'+' is over in %.2f sec...'%(fit_time))
    accumulate_ratio = [sum(data_pca.explained_variance_ratio_[:i])\
                            for i in range(np.shape(X)[1])]
    for i in accumulate_ratio:
        if i > dr_thrs:
            n_select = accumulate_ratio.index(i)
            break
    print('Select %s dims %.5f percent variance explained' % (n_select, 100*dr_thrs))
    matplotlib.rcParams['figure.figsize'] = (3, 3)
    plt.plot(accumulate_ratio)
    plt.axhline(y=dr_thrs, color='g')
    plt.axvline(x=n_select,linewidth=2,color='k')
    plt.axis([0, n_select*2, np.mean(accumulate_ratio), 1])
    plt.title("Percentage of variance explained")
    plt.show()
    X_select = PCA(n_select).fit_transform(X)
    return X_select
def test_ReDim(X_rd, X, y, _clf = RandomForestRegressor()):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    # no pca
    clf = _clf
    start_time = time.time()
    clf.fit(X_train, y_train)
    print('clf'+' has a rmse_cv_mean of %.4f...'%(rmse_cv_mean(clf, X, y)))
    total_time = time.time() - start_time
    print('clf'+' is over in %.2f sec...'%(total_time))
    # with pca
    clf = _clf
    X_train, X_test, y_train, y_test = train_test_split(X_rd, y, test_size=.4, random_state=42)
    start_time = time.time()
    clf.fit(X_train, y_train)
    print('(PCA) clf'+' has a rmse_cv_mean of %.4f...'%(rmse_cv_mean(clf, X, y)))
    total_time = time.time() - start_time
    print('(PCA) clf'+' is over in %.2f sec...'%(total_time))
    print()
    return X_rd
thrs = 1-1e-4
X_rd = test_ReDim(reduce_dim(X, thrs), X, y)
test_rd = PCA(np.shape(X_rd)[1]).fit_transform(test_data)
#, LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

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
coeff_imp_vis()

# using pca
#%% models
predictions = dict()
scores = dict()
time_costs = dict()
test_preds = dict()

#CV_params = list(np.logspace(-5, 1))
CV_params = [10, 1, 0.1, 0.001, 0.0005]
models = {
#    "ARDRegression": ARDRegression(), #.13 slow
#    "HuberRegressor": HuberRegressor(), #.29ERROR pca
#    "TheilSenRegressor": TheilSenRegressor(), #. slow
#    "SGDRegressor": SGDRegressor(), #inf cannot convergence
#    "RadiusNeighborsRegressor": RadiusNeighborsRegressor(), #ERROR
#    "GaussianProcessRegressor": GaussianProcessRegressor(), #12.01
#    "Neural Net": MLPRegressor(), #1.24
#    "PassiveAggressiveClassifier": PassiveAggressiveRegressor(), #.36
#    "SVR": SVR(), #.26
#    "Linear SVR": LinearSVR(), #.29
#    "DecisionTree": DecisionTreeRegressor(), #.20
#    "KNeighborsRegressor": KNeighborsRegressor(), #.26
#    "RANSACRegressor": RANSACRegressor(), #.18
#    "Linear Regression": LinearRegression(), #.17

#    "LarsCV": LarsCV(), #.14
#    "BayesianRidge": BayesianRidge(), #.1277
    "LassoLarsCV": LassoLarsCV(max_iter=2000), #.1258
#    "RidgeCV": RidgeCV(alphas=CV_params), #.1273
    "ElasticNetCV": ElasticNetCV(alphas=CV_params), #.1229
    "LassoCV": LassoCV(alphas=CV_params), #.1231

#    "AdaBoostRegressor": AdaBoostRegressor(), #.17
#    "BaggingRegressor": BaggingRegressor(), #.15
#    "ExtraTreesRegressor": ExtraTreesRegressor(), #.15
#    "Random Forest": RandomForestRegressor(), #.15
    "GradientBoostingRegressor": GradientBoostingRegressor(), #.1262 
    }

X_train, X_test, y_train, y_test = train_test_split(X_rd, y, test_size=.2, random_state=22)
# no pca
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

cnt = 0
for name, model in models.items():
    cnt += 1
    print('\nTraining '+name+' No.%d CLF in all...'%(cnt))
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(name+' is trained in %.4f sec No.%d in all...'%(fit_time,cnt))
    print('Prediction of '+name+' No.%d CLF in all...'%(cnt))
    start_time = time.time()
    pred = model.predict(X_test)
    predict_time = time.time() - start_time
    time_costs[name] = [fit_time, predict_time]
    predictions[name] = pred
    test_preds[name] = model.predict(test_rd)
    print(name+' predicted in %.4f sec No.%d in all...'%(predict_time,cnt))
    scores[name] = rmse_cv_mean(model, X, y)
    print(name+' has a score of %.4f No.%d in all...'%(scores[name],cnt))

#%% Ensemble by simple averaging
averaging = np.zeros((len(y_test),))
averaging_weighted = np.zeros((len(y_test),))
test_ave = np.zeros((np.shape(test_data.ix[:,0])))
test_ave_w = np.zeros((np.shape(test_data.ix[:,0])))
weight_sum = 0
[(name, np.exp(.12-weight)) for name, weight in list(scores.items())]
score_weights = dict()
for name, pred in predictions.items():
    weight = 1 / scores[name] / weight_sum
    score_weights[name] = weight
    averaging_weighted += pred * weight
    averaging += pred / cnt
    test_ave += test_preds[name] / cnt# predict-ave on testdata
    test_ave_w += test_preds[name] * weight
test_preds['Averaging'] = test_ave
test_preds['Averaging_weighted'] = test_ave_w

scores['Averaging']  = rmse(y_test, averaging)
scores['Averaging_weighted']  = rmse(y_test, averaging_weighted)
print('Averaging'+' has a score of'+'\nAveraging: %.6f'%(scores['Averaging']))
print('Averaging_weighted'+' has a score of'+'\nAveraging_weighted: %.6f'%
      (scores['Averaging_weighted']))

#%% Ensemble by stacking
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, random_state=41)

stack_models = {
    "LassoLarsCV": LassoLarsCV(max_iter=2000), #.1258
    "ElasticNetCV": ElasticNetCV(alphas=CV_params), #.1229
    "LassoCV": LassoCV(alphas=CV_params), #.1231
    "GradientBoostingRegressor": GradientBoostingRegressor(), #.1262 
    }
stack_trainSet = []
stack_scores = dict()
stack_preds = dict()

#X_train, X_test, y_train, y_test = train_test_split(X_rd, y, test_size=.1, random_state=22)
# no pca
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

foldI = 0
for train_index, test_index in skf.split(X_rd, y):
    X_train, y_train = X_rd[train_index], y[train_index]
    X_test, y_test = X_rd[test_index], y[test_index]
    stack_trainSet += [np.reshape(y_test,(len(y_test),1))]
    pred_set = []
    print('\n---No.%s fold of %s in all---' % (foldI, 3))
    cnt = 0
    for name, model in stack_models.items():
        cnt += 1
        print('\nTraining '+name+' No.%d CLF in all...'%(cnt))
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        print(name+' is trained in %.4f sec No.%d in all...'%(fit_time,cnt))
        print('Prediction of '+name+' No.%d CLF in all...'%(cnt))
        weight = score_weights[name] #
        start_time = time.time()
        pred = model.predict(X_test) * weight
        predict_time = time.time() - start_time
        time_costs[name] = [fit_time, predict_time]
        stack_preds[name] = pred
        print(name+' predicted in %.4f sec No.%d in all...'%(predict_time,cnt))
        stack_scores[name] = rmse_cv_mean(model, X, y)
        print(name+' has a score of %.4f No.%d in all...'%(scores[name],cnt))
        pred_set += [np.reshape(pred,(len(pred),1))]
    for pred in pred_set:
        pred = np.reshape(pred,(len(pred),1))
        stack_trainSet[foldI-1] = np.hstack((stack_trainSet[foldI-1], pred))

stack_train = stack_trainSet[0]
for train in stack_trainSet[1:]:
    stack_train = np.vstack((stack_train, train))

from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler

stack_train_ = stack_train[:,1:]
plt.hist(stack_train_)
stack_train_ = StandardScaler().fit_transform(stack_train[:,1:])
plt.hist(stack_train_)
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
        stack_train_, stack_train[:,0], test_size=.1, random_state=22)

print(stack_scores)

meta_models = {
    "LassoLarsCV": LassoLarsCV(max_iter=2000), #.1258
    "ElasticNetCV": ElasticNetCV(alphas=CV_params), #.1229
    "LassoCV": LassoCV(alphas=CV_params), #.1231
    "GradientBoostingRegressor": GradientBoostingRegressor(), #.1262 
    }
meta_scores = dict()
cnt = 0
for name, model in meta_models.items():
    cnt += 1
    model.fit(X_train, y_train)
    meta_scores[name] = rmse_cv_mean(model, X_test, y_test)
    print(name+' has a score of %.4f No.%d in all...'%(meta_scores[name],cnt))

print('single_model scores')
print(sorted(stack_scores.values())[0])
print('stack_meta scores')
print(sorted(meta_scores.values())[0])

scores['Stacking']  = sorted(meta_scores.values())[0]

#%% stacking test
stack_testSet = []
test_y = np.zeros((np.shape(test_rd)[0],))
foldI = 0
weight_sum = 0
cnt = 0
for name, model in stack_models.items():
    cnt += 1
    weight = score_weights[name]
    pred = model.predict(test_rd) * weight
    stack_testSet.append(pred)
    
stack_test = stack_testSet[0]
for test in stack_testSet[1:]:
    stack_test = np.vstack((stack_test, test))
stack_test = stack_test.T
meta_name = min(meta_scores.items(), key=lambda x: x[1])[0]
test_preds['Stacking'] = meta_models[meta_name].predict(stack_test)

