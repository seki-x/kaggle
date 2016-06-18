# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 17:42:27 2016

@author: john
"""

import pandas as pd
import numpy as np

def loadData(fileName):
    '''
    Read data from file
    '''
    data = pd.read_csv(fileName,infer_datetime_format=True)
    return data

train = loadData('./data/train.csv')
test = loadData('./data/test.csv')

def byTimeIndex(df):
    df_by_dt = df.sort_values(by='DateTime')
    df_by_dt['time_index'] = range(0,len(df))
    df_by_dt = df_by_dt.reindex(index=df.index)
    df_by_dt = df_by_dt.drop(['DateTime'],axis=1)
    return df_by_dt

def mapName(df):
    namemap = {}
    def mapName2Num(nm):
        if nm in namemap:
            namemap[nm] += 1
        else:
            namemap[nm] = 1
        return namemap[nm]
    
    df['namefq'] = df['Name'].map(mapName2Num)
    df = df.drop(['Name'],axis=1)
    return df

#==============================================================================
# OutcomeType 
#1     Adoption       :10769
#5     Died           :  197
#4     Euthanasia     : 1555
#3     Return_to_owner: 4786
#2     Transfer       : 9422
#==============================================================================

def mapOCtype(df):
    ocMap = {'Adoption':1,
             'Transfer':5,
             'Return_to_owner':4,
             'Euthanasia':3,
             'Died':2
    }
    df['octype'] = df['OutcomeType'].map(ocMap)
    df = df.drop(['OutcomeType'],axis=1)
    return df

#==============================================================================
# def mapSubtype(df):
#     ocSubmap = {}
#     def mapSubtype2Num(subtype):
#         if subtype in ocSubmap:
#             ocSubmap[subtype] += 1
#         else:
#             ocSubmap[subtype] = 1
#         return ocSubmap[subtype]
#     
#     df['ocsubtypefq'] = df['OutcomeSubtype'].map(mapSubtype2Num)
#     df = df.drop(['OutcomeSubtype'],axis=1)
#     return df
#==============================================================================

def mapAnimalType(df):
    isCatMap = {'Cat':1,
             'Dog':0,
    }
    df['iscat'] = df['AnimalType'].map(isCatMap)
    df = df.drop(['AnimalType'],axis=1)
    return df

def mapSex(df):
    sex = {'Neutered Male':1,
           'Spayed Female':2,
           'Intact Male':3,
           'Intact Female':4,
           'Unknown':5
    }
    df['sex'] = df['SexuponOutcome'].map(sex)
    df = df.drop(['SexuponOutcome'],axis=1)
    return df

def mapColor(df):
    colormap = {}
    def mapColor2Num(cl):
        if cl in colormap:
            colormap[cl] += 1
        else:
            colormap[cl] = 1
        return colormap[cl]
    
    df['colorfq'] = df['Color'].map(mapColor2Num)
    df = df.drop(['Color'],axis=1)
    return df

def mapBreed(df):
    breedmap = {}
    def mapBreed2Num(br):
        if br in breedmap:
            breedmap[br] += 1
        else:
            breedmap[br] = 1
        return breedmap[br]
    
    df['breedfq'] = df['Breed'].map(mapBreed2Num)
    df = df.drop(['Breed'],axis=1)
    return df

def mapAge(df):
    def mapAge2Days(age):
        [num,unit] = age.split(' ')
        num = int(num)
        if unit[-1]=='s':
            unit = unit[:len(unit)-1]
        mapUnit = {'day':1,
                   'week':7,
                   'month':30,
                   'year':365
        }
        ageDays = num * mapUnit[unit]        
        return ageDays        
    df['AgeuponOutcome'] = df['AgeuponOutcome'].fillna('0 days')    
    df['agedays'] = df['AgeuponOutcome'].map(mapAge2Days)
    df = df.drop(['AgeuponOutcome'],axis=1)
    return df

Outcome = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']


def dataProcTrain(t):
    t = byTimeIndex(t)
    t = mapName(t)
    t = mapOCtype(t)
    t = mapAnimalType(t)
    t = mapSex(t)
    t = mapColor(t)
    t = mapBreed(t)
    t = mapAge(t)
    return t

def dataProcTest(t):
    t = byTimeIndex(t)
    t = mapName(t)
    t = mapAnimalType(t)
    t = mapSex(t)
    t = mapColor(t)
    t = mapBreed(t)
    t = mapAge(t)
    return t

sp = train.drop(['AnimalID','OutcomeSubtype'],axis=1)
train = dataProcTrain(sp)
octype = train.pop('octype')
train.insert(0,'octype',octype)
idx = test['ID']
sp = test.drop(['ID'],axis=1)
test = dataProcTest(sp)
del sp
#--------------------------- data cleaning end

train_data = train.values.astype(np.int64)
test_data = test.values

#def scaling(col):

print 'Training...'

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

forest0 = forest.fit(train_data[0:10000,1::],train_data[0:10000,0])
#output0 = forest.predict(train_data[20001:,1::])
output0 = forest.predict(train_data[20001:,1::])
true0 = train_data[20001:,0]

def calcError(res, answer):
    '''
    Calculate error ration
    '''
    ER = 0.0
    for idx in range(0, len(answer)):
        if res[idx] != answer[idx]:
            ER += 1.0
    ER = ER / float(len(answer))
    return ER

res0 = calcError(true0,output0)

forest = forest.fit(train_data[0:,1:], train_data[0:,0])
output = forest.predict(test_data[0:,0:])

#--------------------------- prediction end

import csv as csv
predictions_file = open('shelter0.csv', 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer'])
outputdict = {
    1:'1,0,0,0,0',
    2:'0,1,0,0,0',
    3:'0,0,1,0,0',
    4:'0,0,0,1,0',
    5:'0,0,0,0,1',
}
out = [outputdict[x] for x in output]
open_file_object.writerows(zip(idx, out))
predictions_file.close()
print 'Done.'

#--------------------------- writing csv end



