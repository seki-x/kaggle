# -*- coding: utf-8 -*-
'''
Created on Wed Jun 15 18:26:43 2016

@author: john
'''
import pandas as pd
import numpy as np

def loadData(fileName):
    '''
    Read data from file
    '''
    data = pd.read_csv(fileName)
    return data

#fileName = './data/data.csv'
data = loadData('./data/data.csv')
data.shot_made_flag = data.shot_made_flag.fillna(2)
shot_made_flag = data.pop('shot_made_flag')
data.insert(0, 'shot_made_flag', shot_made_flag)
train = data[data['shot_made_flag']!=2].drop\
    (['game_event_id','game_id','lat','lon','team_name','team_id','shot_id'],axis=1)
test = data[data['shot_made_flag']==2].drop\
    (['game_event_id','game_id','lat','lon','team_name','team_id','shot_id','shot_made_flag'],axis=1)


def addMatchType(df):
    ## away or home (1 or 0)
    def mapMatchType(matchup):
        if '@' in matchup:
            return 1 #'away'
        else:
            return 0 #'home'
    df['IsMatchAway'] = df['matchup'].map(mapMatchType)
    return df

train = addMatchType(train).drop(['matchup'],axis=1)
test = addMatchType(test).drop(['matchup'],axis=1)

# combined shot type
#------------------------ 1~6 or 141 - 23485(freq) ???
#    1    Bank Shot:  141
#    4    Dunk     : 1286
#    2    Hook Shot:  153
#    6    Jump Shot:23485
#    5    Layup    : 5448
#    3    Tip Shot :  184

def mapCombShotType(df):
    mapCombinedShotType = {'Bank Shot':1,
           'Dunk':2,
           'Hook Shot':3,
           'Jump Shot':4,
           'Layup':5,
           'Tip Shot':6}
    df['CombinedShotType'] = df['combined_shot_type']\
    .map(mapCombinedShotType)
    df = df.drop(['action_type','combined_shot_type'],axis=1)
    return df

train = mapCombShotType(train)
test = mapCombShotType(test)

# season
#    1    1996-97
#    .    ...
#    4    2000-01
#    5    2001-02
#    .    ...
def addSeasonNum(df):
    def mapSeason2Num(season):
        num = int(season[-2:])
        if num > 96:
            return (num - 96)
        else:
            return (num + 4)
    df['SeasonNum'] = df['season'].map(mapSeason2Num)
    df = df.drop(['season'],axis=1)
    return df

train = addSeasonNum(train)
test = addSeasonNum(test)

# shot type
#    0    2PT Field Goal:24271
#    1    3PT Field Goal: 6426

def addIsShotType3PT(df):
    mapShotType3PT = {'2PT Field Goal': 0, 
                '3PT Field Goal': 1}
    df['IsShotType3PT'] = df['shot_type'].map(mapShotType3PT)
    df = df.drop(['shot_type'],axis=1)
    return df
    
train = addIsShotType3PT(train)
test = addIsShotType3PT(test)

# shot_zone_area
#    1    Back Court(BC)       :   83
#    6    Center(C)            :13455
#    3    Left Side Center(LC) : 4044
#    2    Left Side(L)         : 3751
#    4    Right Side Center(RC): 4776
#    5    Right Side(R)        : 4588

def mapShotZoneArea(df):
    mapShotZoneArea2Num = {'Back Court(BC)':1,
           'Left Side(L)':2,
           'Left Side Center(LC)':3,
           'Right Side Center(RC)':4,
           'Right Side(R)':5,
           'Center(C)':6}
    df['ShotZoneAreaNum'] = df['shot_zone_area'].map(mapShotZoneArea2Num)
    df = df.drop(['shot_zone_area'],axis=1)
    return df

train = mapShotZoneArea(train)
test = mapShotZoneArea(test)

#shot_zone_basic
#    5    Above the Break 3    : 5620
#    1    Backcourt            :   71
#    4    In The Paint (Non-RA): 4578
#    2    Left Corner 3        :  280
#    7    Mid-Range            :12625
#    6    Restricted Area      : 7136
#    3    Right Corner 3       :  387

def mapShotZoneBasic(df):
    mapShotZoneBasic2Num = {'Backcourt':1,
           'Left Corner 3':2,
           'Right Corner 3':3,
           'In The Paint (Non-RA)':4,
           'Above the Break 3':5,
           'Restricted Area':6,
           'Mid-Range':7}
    df['ShotZoneBasicNum'] = df['shot_zone_basic'].map(mapShotZoneBasic2Num)
    df = df.drop(['shot_zone_basic'],axis=1)
    return df

train = mapShotZoneBasic(train)
test = mapShotZoneBasic(test)

#shot_zone_range
#    4    16-24 ft.      :8315 
#    2    24+ ft.        :6275 
#    3    8-16 ft.       :6626 
#    1    Back Court Shot:  83 
#    5    Less Than 8 ft.:9398 

def mapShotZoneRange(df):
    mapShotZoneRange2Num = {'Back Court Shot':1,
           '24+ ft.':2,
           '8-16 ft.':3,
           '16-24 ft.':4,
           'Less Than 8 ft.':5}
    df['ShotZoneRangeNum'] = df['shot_zone_range'].map(mapShotZoneRange2Num)
    df = df.drop(['shot_zone_range'],axis=1)
    return df

train = mapShotZoneRange(train)
test = mapShotZoneRange(test)

#game_date  
#    2016-04-13:   50
#    2002-11-07:   47
#    2006-01-22:   46
#    2006-12-29:   45
#    2007-03-30:   44
#    2008-01-14:   44
#    (Other)   :30421

game_date = data['game_date']

def addDailyShotNum(df):
    dailyShotNum = {}
    def mapGameDay2DailyShotNum(gameDay):
        if gameDay in dailyShotNum:
            dailyShotNum[gameDay] += 1
        else:
            dailyShotNum[gameDay] = 1
        return dailyShotNum[gameDay]
    
    df['DailyShotNum'] = df['game_date'].map(mapGameDay2DailyShotNum)
    df = df.drop(['game_date'],axis=1)
    return df

train = addDailyShotNum(train)
test = addDailyShotNum(test)

#opponent  
#    SAS    : 1978
#    PHX    : 1781
#    HOU    : 1666
#    SAC    : 1643
#    DEN    : 1642
#    POR    : 1539
#    (Other):20448

def mapOpponentNum(df):
    opList = list(np.unique(df['opponent']))
    opDict = {}
    for idx, op in enumerate(opList):
        opDict[op] = idx + 1
    def mapOpponent2Num(op):
        return opDict[op]
    df['OpponentNum'] = df['opponent'].map(mapOpponent2Num)
    df = df.drop(['opponent'],axis=1)
    return df

train = mapOpponentNum(train)
test = mapOpponentNum(test)

#--------------------------- data cleaning end

train_data = train.values
test_data = test.values

print 'Training...'

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
#Calc logloss
def logloss(act, pred):
    import scipy as sp
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

forest0 = forest.fit( train_data[0:20000,1::], train_data[0:20000,0] )
#output0 = forest.predict(train_data[20001:,1::])
output0 = forest.predict_proba(train_data[20001:,1::])[:,1]

print 'logloss...'
print(logloss(train_data[20001:,0], output0))

forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
print 'Predicting...'
output = forest.predict_proba(test_data)[:,1]

#--------------------------- prediction end

import csv as csv
predictions_file = open('kobeprob.csv', 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['shot_id','shot_made_flag'])
open_file_object.writerows(zip(test.index+1, output))
predictions_file.close()
print 'Done.'

#--------------------------- writing csv end
