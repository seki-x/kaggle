# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 17:17:45 2016

@author: john
"""
import pandas as pd
import numpy as np

def loadData(fname):
    df = pd.read_csv('data/'+fname, header=0)
    df['Gender'] = 4
    df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & \
                                  (df['Pclass'] == j+1)]['Age'].dropna().median()
    df['AgeFill'] = df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                    'AgeFill'] = median_ages[i,j]
    del i, j
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass
    df.dtypes[df.dtypes.map(lambda x: x=='object')]
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }
    df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
    
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)
    median_age = df['Age'].dropna().median()
    if len(df.Age[ df.Age.isnull() ]) > 0:
        df.loc[ (df.Age.isnull()), 'Age'] = median_age
    ids = df['PassengerId'].values
    df = df.drop(['PassengerId'], axis=1)
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
    data = df.values
    return ids, data

_, train_data = loadData('train.csv')
ids, test_data = loadData('test.csv')

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
output = forest.predict(test_data).astype(int)

import csv
predictions_file = open("myforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
