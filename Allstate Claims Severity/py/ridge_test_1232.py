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

y_loss_1_log1p = pd.DataFrame({"loss":train["loss"], \
                       "log(loss + 1)":np.log1p(train["loss"])})
y_loss_1_log1p.hist()


from scipy.stats import skew
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = np.log1p(train["loss"])

#%% Models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def mae_cv(model):
    mae= -cross_val_score(model, X_train, y, scoring="mean_absolute_error", cv = 5)
    return(mae)

model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]#
cv_ridge = [mae_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("mae")
cv_ridge.min()

alphas = pd.Series(alphas)
alpha_best = cv_ridge.idxmin()
ridge_best = Ridge(alpha = alpha_best).fit(X_train,y)
pred = np.expm1(ridge_best.predict(X_test))

#solution = pd.DataFrame({"id":test_id, "loss":pred})
#solution.to_csv("ridge_sol.csv", index = False)
