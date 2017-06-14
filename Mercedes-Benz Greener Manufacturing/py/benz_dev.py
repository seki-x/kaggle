# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:04:42 2017

@author: john
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df_train = pd.read_csv(r'../input/train.csv')
df_test = pd.read_csv(r'../input/test.csv')

target = df_train['y']
target.hist(); plt.show()
target_log1p = pd.Series(np.log1p(target))
target_log1p.hist(); plt.show()

train_len = len(df_train)

target = df_train['y']
target_log1p = pd.Series(np.log1p(target))
target_log1p.hist(); plt.show()

df_train = df_train.drop(['y'], axis=1)
all_data = pd.concat((df_train, df_test), axis=0, ignore_index=True)
all_data['ID'].plot(); plt.show()
all_data = all_data.drop(['ID'], axis=1)
#%%
obj_feats = []
non_obj_feats = []
for col in all_data.columns:
    obj_feats.append(col) if all_data[col].dtype == 'O' else non_obj_feats.append(col)
print('Object features')
print(obj_feats)
print('non-Object features')
print(non_obj_feats)

df_obj_feats = all_data[obj_feats]
df_non_obj_feats = all_data[non_obj_feats]
#%%
from collections import Counter
uniVal_len = []
uniVals = dict()
df_obj_feats_freq = pd.DataFrame()
map_to_cnt = lambda k: uniVals[col][k]
map_to_val = lambda s: ord(s) - ord('a') + 1
map_to_sum = lambda k: sum([map_to_val(s) for s in k])
map_to_len = lambda k: len(k)
for col in df_obj_feats.columns:
    print('column:',col)
    uniVal = np.unique(df_obj_feats[col])
    print(uniVal)
    uniVal_len.append(len(uniVal))
    print('unique value numbers:', len(uniVal))
    uniVals[col] = Counter(df_obj_feats[col])
    df_obj_feats_freq[col] = df_obj_feats[col].apply(map_to_cnt)
    df_obj_feats_freq[col] /= len(df_obj_feats[col])
    df_obj_feats_freq[col+'_len'] = df_obj_feats[col].apply(map_to_len)
    df_obj_feats_freq[col+'_sum'] = df_obj_feats[col].apply(map_to_sum)
    df_obj_feats_freq[col+'_sum'] += 26 * df_obj_feats_freq[col+'_len']
uniVal_len = pd.DataFrame(uniVal_len, df_obj_feats.columns)
uniVal_len.plot(); plt.show()

#%%
drop_cols_train = []
for col in df_train.columns:
    if len(df_train[col].unique()) == 1:
        drop_cols_train.append(col)
print('Non-unique values columns:', drop_cols_train)
df_non_obj_feats = df_non_obj_feats.drop(drop_cols_train, axis=1)
drop_cols_obj = []
for col in df_obj_feats_freq.columns:
    if len(df_obj_feats_freq[col].unique()) == 1:
        drop_cols_obj.append(col)
print('Non-unique values columns:', drop_cols_obj)
df_obj_feats_freq = df_obj_feats_freq.drop(drop_cols_obj, axis=1)

#%%
import sklearn.linear_model as sk_lm
import sklearn.ensemble as sk_ens
import xgboost as xgb
import sklearn.neural_network as sk_nn

from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, KFold

df_X_train = df_non_obj_feats[:train_len]
df_X_test = df_non_obj_feats[train_len:]
y = target_log1p

df_importance = pd.DataFrame(data=None, index=df_non_obj_feats.columns)
#%%
model_lasso = sk_lm.LassoCV(alphas=[3e-4, 3e-3, 3e-2, 3e-1, 3, 30])
model_lasso.fit(df_X_train, y)
model_lasso.score(df_X_train, y)
r2_score_lasso = r2_score(y, model_lasso.predict(df_X_train))
print('r2_score of Lasso:', r2_score_lasso)
lasso_importance = pd.DataFrame(model_lasso.coef_,
                             df_X_train.columns,['LS_feat_importance'])
lasso_importance.plot(); plt.show()

df_importance['LassoCV'] = r2_score_lasso*lasso_importance/np.max(np.abs(lasso_importance))
#%%
model_elen = sk_lm.ElasticNetCV(alphas=[3e-4, 3e-3, 3e-2, 3e-1, 3, 30])
model_elen.fit(df_X_train, y)
model_elen.score(df_X_train, y)
r2_score_elen = r2_score(y, model_elen.predict(df_X_train))
print('r2_score of ElasticNet:', r2_score_elen)
elen_importance = pd.DataFrame(model_elen.coef_,
                             df_X_train.columns,['ELEN_feat_importance'])
elen_importance.plot(); plt.show()
df_importance['ElasticNet'] = r2_score_elen*elen_importance/np.max(np.abs(elen_importance))
#%%
model_ll = sk_lm.LassoLarsCV()
model_ll.fit(df_X_train, y)
model_ll.score(df_X_train, y)
r2_score_ll = r2_score(y, model_ll.predict(df_X_train))
print('r2_score of LassoLars:', r2_score_ll)
ll_importance = pd.DataFrame(model_ll.coef_,
                             df_X_train.columns,['LL_feat_importance'])
ll_importance.plot(); plt.show()
df_importance['LassoLars'] = r2_score_ll*ll_importance/np.max(np.abs(ll_importance))
#%%
model_rf = sk_ens.RandomForestRegressor()
model_rf.fit(df_X_train, y)
model_rf.score(df_X_train, y)
r2_score_rf = r2_score(y, model_rf.predict(df_X_train))
print('r2_score of RandomForest:', r2_score_rf)
rf_importance = pd.DataFrame(model_rf.feature_importances_,
                             df_X_train.columns,['RF_feat_importance'])
rf_importance.plot(); plt.show()
df_importance['RandomForest'] = r2_score_rf*rf_importance/np.max(np.abs(rf_importance))

#%%
df_importance['all'] = np.sum(df_importance.values, 1)
imp_all = df_importance['all'].sort_values(ascending=False)
imp_all.plot(); plt.show()
df_non_obj_feats_sort = df_non_obj_feats.copy().reindex_axis(imp_all.index, axis=1)
map_bin_dec = lambda row: int('0b'+''.join(list(map(str, list(row)))),2)

bin_feats = df_non_obj_feats_sort.values
int10 = np.array(list(map(map_bin_dec, bin_feats)))
int10 = int10 / max(int10)

df_non_obj_feats['binSum'] = df_non_obj_feats.apply(sum, 1)
df_non_obj_feats['binDec'] = int10

all_data_proc = pd.concat((df_obj_feats_freq, df_non_obj_feats), axis=1)

#%%
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
n_comp = 12

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results = grp.fit_transform(all_data_proc)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results = srp.fit_transform(all_data_proc)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca_results = pca.fit_transform(all_data_proc)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica_results = ica.fit_transform(all_data_proc)
for i in range(1, n_comp+1):
    all_data_proc['pca_' + str(i)] = pca_results[:,i-1]
    all_data_proc['ica_' + str(i)] = ica_results[:, i-1]
    all_data_proc['grp_' + str(i)] = grp_results[:,i-1]
    all_data_proc['srp_' + str(i)] = srp_results[:, i-1]

df_X_train = all_data_proc[:train_len]
df_X_test = all_data_proc[train_len:]
y = target_log1p
#%% CV
X = df_X_train.values
test_data = df_X_test.values
y = target_log1p.values
y_mean = np.mean(y)

gen_rand = lambda : np.random.randint(2333333)

rf_params = dict()
rf_params['n_estimators'] = 1 + int(np.shape(X)[1] ** 0.7)
rf_params['max_features'] = 0.6
rf_params['max_depth'] = 5

xgb_params = dict()
xgb_params['n_estimators'] = 500
xgb_params['learning_rate'] = 0.005
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.921
xgb_params['objective'] = 'reg:linear'
#xgb_params['eval_metric'] = 'rmse'
xgb_params['base_score'] = y_mean
xgb_params['silent'] = 1
xgb_params['seed'] = 233

def CV_mean(X_slct, y, test_slct, model_name='RandomForest',
            model_obj=sk_ens.RandomForestRegressor, model_params=rf_params, 
            eval_func=r2_score, nFolds=5, gen_rand_func=gen_rand):
    k_fold = KFold(n_splits=nFolds, shuffle=True, random_state=gen_rand_func())
    cv_scores = []
    model_li = []
    preds = []
    for train_index, test_index in k_fold.split(X_slct, y):
        X_train, X_test = X_slct[train_index,:], X_slct[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        if 'random_state' in model_params:
            model_params['random_state'] = gen_rand_func()
        elif 'seed' in model_params:
            model_params['seed'] = gen_rand_func()
        model = model_obj(**model_params)
        model.fit(X_train, y_train)
        scr = eval_func(y_test, model.predict(X_test))
        print('Score of ' + model_name + ':', scr)
        model_li.append(model)
        cv_scores.append(scr)
        pred = model.predict(test_slct)
        preds.append(pred)
    plt.plot(cv_scores); plt.show()
    winner_pred = preds[cv_scores.index(max(cv_scores))]
    print('CV_mean ' + model_name + ':', np.mean(cv_scores))
    return np.mean(cv_scores), winner_pred

rf_scr, rf_pred = CV_mean(X, y, test_data)
xgb_scr, xgb_pred = CV_mean(X, y, test_data, 'xgb', xgb.XGBRegressor, xgb_params)

#%%
rf_ratio_flt = rf_scr / (xgb_scr+ rf_scr)
#rf_ratio_flt = 1
rf_ratio = float('{:.2f}'.format(rf_ratio_flt))
xgb_ratio = round(1 - rf_ratio, 2)
pred_final_log1p = rf_ratio * rf_pred + xgb_ratio * xgb_pred
pred_final = np.expm1(pred_final_log1p)
sub = pd.read_csv(r'../input/sample_submission.csv')
sub['y'] = pred_final

#%%
sub.to_csv(r'sub_freq_kSum_dbl_binDecSum_PCAICAGRPSRP'+\
           str(rf_ratio)+'rf'+str(xgb_ratio)+'xgb'+ '.csv', index=False)
