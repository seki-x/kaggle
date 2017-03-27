# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:24:36 2017

@author: John
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
#%% Preprocessing
def prepare_data(fname):
    data_loaded = pd.read_csv(r'.\..\input'+'\\'+fname+'.csv')
    from sklearn.preprocessing import MinMaxScaler
    if fname=='train':
        label_values = data_loaded['label'].values
        feature = data_loaded.iloc[:,1:]
        feature_scaled_values = MinMaxScaler().fit_transform(feature)
        label_values = np.reshape((label_values),(len(label_values),1))
        data_scaled_values = np.concatenate((label_values,\
                                             feature_scaled_values),axis=1)
    if fname=='test':
        feature = data_loaded
        feature_scaled_values = MinMaxScaler().fit_transform(feature)
        data_scaled_values = feature_scaled_values
    return data_scaled_values
def get_train_test(X, y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=.4, random_state=42)


#%% classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression,\
 RidgeClassifier, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
#from sklearn.neighbors import !!! too slow
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.decomposition import PCA


#%% data
data = prepare_data('train')
X = data[:,1:]
y = data[:,0]


#%% dimention reduction
start_time = time.time()
data_pca = PCA(np.shape(X)[1]).fit(X)
fit_time = time.time() - start_time
print('PCA'+' is over in %.2f sec...'%(fit_time))
accumulate_ratio = [sum(data_pca.explained_variance_ratio_[:i])\
                        for i in range(np.shape(X)[1])]
dr_thrs = .90
for i in accumulate_ratio:
    if i > dr_thrs:
        n_select = accumulate_ratio.index(i)
        break
plt.plot(accumulate_ratio)
plt.axhline(y=dr_thrs, color='g')
plt.axvline(x=n_select,linewidth=4,color='k')
plt.axis([0, n_select*2, 0, 1])
plt.show()
plt.close()

#%% test single 
X_train, X_test, y_train, y_test = get_train_test(X, y)
# no pca
clf = LogisticRegression()
start_time = time.time()
clf.fit(X_train, y_train)
print('clf'+' has a score of %.2f...'%(clf.score(X_test, y_test)))
total_time = time.time() - start_time
print('clf'+' is over in %.2f sec...'%(total_time))
# with pca
clf = LogisticRegression()
X_select = PCA(n_select).fit_transform(X)
X_train, X_test, y_train, y_test = get_train_test(X_select, y)
start_time = time.time()
clf.fit(X_train, y_train)
print('clf'+' has a score of %.2f...'%(clf.score(X_test, y_test)))
total_time = time.time() - start_time
print('(PCA) clf'+' is over in %.2f sec...'%(total_time))


# no pca
X_train, X_test, y_train, y_test = get_train_test(X, y)

#%% Single model
predictions = dict()
scores = dict()
time_costs = dict()
classifiers = {
    "Linear SVM": LinearSVC(), #.93
    "Neural Net": MLPClassifier(), # .99 s 1.5min
    "Logistic Regression": LogisticRegression(), #.93 s 1min
    "Random Forest": RandomForestClassifier(), #.93
    "RidgeClassifier": RidgeClassifier(), # .86
    "SGD": SGDClassifier(), # .89 
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(), #.88
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier(), #.86
    "DecisionTreeClassifier": DecisionTreeClassifier(), #.85
#    "BernoulliNB": BernoulliNB(), # X must be pos .83 
#    "MultinomialNB": MultinomialNB() # X must be pos .82 
    
    }

cnt = 0
for name, clf in classifiers.items():
    cnt += 1
    print('\nTraining '+name+' No.%d CLF in all...'%(cnt))
    start_time = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(name+' is trained in %.2f sec No.%d in all...'%(fit_time,cnt))
    print('Prediction of '+name+' No.%d CLF in all...'%(cnt))
    start_time = time.time()
    pred = clf.predict(X_test)
    predict_time = time.time() - start_time
    predictions[name] = pred
    print(name+' predicted in %.2f sec No.%d in all...'%(predict_time,cnt))
    scores[name] = sum(y_test==pred)/len(pred)
    print(name+' has a score of %.2f No.%d in all...'%(scores[name],cnt))
    time_costs[name] = [fit_time, predict_time]
    print('Over training %s No.%d in all...'%(name,cnt))
    
#%% Ensemble by simple voting
from sklearn.preprocessing import label_binarize
pred_voting = np.zeros((len(y_test),))
votes = np.zeros((len(y_test),10))
for name, pred in predictions.items():
    vote = label_binarize(pred, list(range(0,10)))
    votes += vote # * np.log(scores[name]/(1-scores[name]))
pred_voting = np.argmax(votes, 1)
scores['Voting']  = sum(y_test==pred_voting)/len(y_test) # .96
print('\nVoting'+' has a score of'+\
      '\nVoting: %.2f' \
      %(scores['Voting']))

#%% Ensemble by stacking
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, random_state=42)

stack_clfs = {
    "Neural Net1": MLPClassifier(), #.97 s 1.5min
    "Neural Net2": MLPClassifier(hidden_layer_sizes=(200,),\
                                 max_iter=300), #.97 s 1.5min
    "Neural Net3": MLPClassifier(hidden_layer_sizes=(300,),\
                                 max_iter=400), #.97 s 1.5min
    "Neural Net4": MLPClassifier(hidden_layer_sizes=(400,),\
                                 max_iter=500), #.97 s 1.5min
#    "Random Forest1": RandomForestClassifier(), #.93
    "Random Forest2": RandomForestClassifier(n_estimators=50), #.93
#    "Logistic Regression": LogisticRegression(), #.93 s 1min
#    "Linear SVM": LinearSVC(), #.93 s 3min
#    "SGD": SGDClassifier(), #.89 
#    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(), #.88
#    "PassiveAggressiveClassifier": PassiveAggressiveClassifier(), #.86
#    "RidgeClassifier": RidgeClassifier(), # .86
#    "DecisionTreeClassifier": DecisionTreeClassifier(), #.85
#    "BernoulliNB": BernoulliNB(), #.83 X must be pos 
#    "MultinomialNB": MultinomialNB() #.82 X must be pos 
    }
stack_names = []
stack_trainSet = []
label_set = []
stack_scores = []

foldI = 0

#for train_index, test_index in skf.split(X_select, y): #with PCA
X_train, X_test, y_train, y_test = get_train_test(X, y)
# no pca
for train_index, test_index in skf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    stack_trainSet += [np.reshape(y_test,(len(y_test),1))]
    pred_set = []
    foldI += 1
    print('\n---No.%s fold of %s in all---' % (foldI, 3))
    cnt = 0
    for name, clf in stack_clfs.items():
        cnt += 1
        print('\nTraining '+name+' No.%d CLF in all...'%(cnt))
        start_time = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - start_time
        print(name+' is trained in %.2f sec No.%d in all...'%(fit_time,cnt))
        print('Prediction of '+name+' No.%d CLF in all...'%(cnt))
        start_time = time.time()
        pred = clf.predict(X_test)
        predict_time = time.time() - start_time
        print(name+' predicted in %.2f sec No.%d in all...'%(predict_time,cnt))
        score = sum(y_test==pred)/len(pred)
        print(name+' has a score of %.2f No.%d in all...'%(score,cnt))
        print('Over training %s No.%d in all...'%(name,cnt))
        stack_names += [name]
        stack_scores += [score]
        pred_set += [np.reshape(pred,(len(pred),1))]
    for pred in pred_set:
        pred = np.reshape(pred,(len(pred),1))
        stack_trainSet[foldI-1] = np.hstack((stack_trainSet[foldI-1], pred))

stack_train = stack_trainSet[0]
for train in stack_trainSet[1:]:
    stack_train = np.vstack((stack_train, train))

X_train, X_test, y_train, y_test = get_train_test(stack_train[:,1:], stack_train[:,0])
metaClf = RandomForestClassifier().fit(X_train, y_train)
scores['Stack'] = metaClf.score(X_test, y_test)
print('Stacking score: ', scores['Stack'])
scores['Stack clfs'] = stack_scores
print('Stack clfs highest score: ', max(stack_scores))
stk_highest_name = stack_names[stack_scores.index(max(stack_scores))]
stk_highest_clf = stack_clfs[stk_highest_name]
print('Stack highest clf name: %s' % (stk_highest_name))

#%% make predictions by stack
# test data
testdata = prepare_data('test')
X = testdata
y = np.zeros(np.size(testdata[:,0]))
X_train, X_test, y_train, y_test = get_train_test(X, y)
# no pca
stack_names = []
stack_testSet = []

foldI = 0
for train_index, test_index in skf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    stack_testSet += [np.reshape(y_test,(len(y_test),1))]
    pred_set = []
    foldI += 1
    print('\n---No.%s fold of %s in all---' % (foldI, 3))
    cnt = 0
    for name, clf in stack_clfs.items():
        cnt += 1
        print('Prediction of '+name+' No.%d CLF in all...'%(cnt))
        start_time = time.time()
        pred = clf.predict(X_test)
        predict_time = time.time() - start_time
        print(name+' predicted in %.2f sec No.%d in all...'%(predict_time,cnt))
        print('Over testing %s No.%d in all...'%(name,cnt))
        stack_names += [name]
        pred_set += [np.reshape(pred,(len(pred),1))]
    for pred in pred_set:
        pred = np.reshape(pred,(len(pred),1))
        stack_testSet[foldI-1] = np.hstack((stack_testSet[foldI-1], pred))

stack_train = stack_testSet[0]
for train in stack_testSet[1:]:
    stack_train = np.vstack((stack_train, train))

stack_test = stack_train[:,1:]
test_pred = dict()
test_pred['stack'] = metaClf.predict(stack_test).astype(int)
test_pred['stack_highest'] = stk_highest_clf.predict(X).astype(int)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), \
                  "Label": preds}).to_csv(fname, index=False, header=True)
write_preds(test_pred['stack'], 'results '+ 'stack'+'.csv')
write_preds(test_pred['stack_highest'], 'results '+ 'stack_highest'+'.csv')