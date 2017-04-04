# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:49:22 2016

@author: john
"""
#import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def rmse_cv(model, train_data=X_train):
    rmse= np.sqrt(-cross_val_score(model, train_data, y, 
                                   scoring="mean_squared_error", cv = 5))
    return(rmse)

alpha_rf = RandomForestRegressor(n_estimators=500, n_jobs=-1,random_state=42)
alpha_rf = alpha_rf.fit(X_train, y)
alpha_cv = rmse_cv(alpha_rf)
print(alpha_cv)
feat_importance = alpha_rf.feature_importances_
feat_importance_pd = pd.Series(feat_importance, index = X_train.columns)
#feat_importance_pd.plot()
feat_importance_pd.sort_values(axis=0, ascending=False, inplace=True)
#feat_importance_pd.plot()

from sklearn.linear_model import LassoCV
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print(rmse_cv(model_lasso).mean())

beta_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

def feat_sl2(model):
    selected_cv = []
    selected_nums = range(1, int(len(feat_importance_pd.index)))
    for selected_num in selected_nums:
        feat_selected = feat_importance_pd.index[:selected_num]
        X_train_sl = X_train[feat_selected] 
        model_fit = model.fit(X_train_sl, y)
        print(selected_num, rmse_cv(model_fit, X_train_sl))
        selected_cv.append(rmse_cv(model_fit, X_train_sl))
    selected_cv_mean = pd.Series([x.mean() for x in selected_cv], index=selected_nums)
    plt.plot(selected_cv_mean)
    selected_cv_mean.sort_values(axis=0, ascending=True, inplace=True)
    best_num = selected_cv_mean.index[0]
    print(best_num)
    return selected_cv, best_num
        
selected_cv, best_num = feat_sl2(beta_lasso)        

feat_selected = feat_importance_pd.index[:best_num]
X_train_sl = X_train[feat_selected] 
beta_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
beta_lasso = beta_lasso.fit(X_train_sl, y)
X_test_sl = X_test[feat_selected] 
preds = np.expm1(beta_lasso.predict(X_test_sl))
# rflb 0.12147
#solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
#solution.to_csv("lasso_sl168.csv", index = False)



