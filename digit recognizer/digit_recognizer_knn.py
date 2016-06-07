# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 14:53:16 2016

@author: john
"""

def loadData(dataFile):
    '''
    Read data from file
    '''
    with open(dataFile, 'rb') as f:
        lines = f.readline()
        lines = f.readlines()
        data = []
        for line in lines:
            line = [int(x) for x in line.split(',')]
            data.append(line)
    return data

def getDataArr():
    '''
    Load data files and return data in np array type
    '''
    dataDir = 'data//'
    dataFile1 = 'train.csv'
    dataFile2 = 'test.csv'
    trainSet = loadData(dataDir+dataFile1)
    testSet = loadData(dataDir+dataFile2)
    del dataFile1, dataFile2, dataDir
    from numpy import array
    dataArr = array(trainSet)
    testArr = array(testSet)
    del trainSet, testSet
    labelArr = dataArr[:, 0]
    trainArr = dataArr[:, 1:]
    return labelArr, trainArr, testArr

def getSubSample(trNumber, trSize, teNumber):
    '''
    Get sub set and sample set to be classified
    '''
    trSample = train[:trNumber]
    labSample = label[:trNumber]
    trSub = train[trainNumber:trNumber + trSize]
    labSub = label[trainNumber:trNumber + trSize]
    teSample = test[:teNumber]
    return trSample, labSample, trSub, labSub, teSample

def saveResult(res, fname):
    '''
	Save result to [fname].csv
	'''
    with open(fname+'.csv', 'w+') as f:
        f.write('ImageId,Label\n')
        for i, resi in enumerate(res):
            f.write(str(i+1)+','+str(resi)+'\n')

def kNNClassify(inX, k=3):
    '''
	kNN classify function in the book 'Machine Learning in Action' chapter 1
	'''
    dataSet = trainSub
    labelSet = labelSub
    import operator
    from numpy import tile
    dataSetSize = dataSet.shape[0]
    sqDiffMat = (tile(inX, (dataSetSize, 1))-dataSet) ** 2
    distances = sqDiffMat.sum(axis=1) ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labelSet[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+ 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

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

def initOption(_trainNumber, _trainSize, _testNumber):
    '''
    Initialize the test and train options
    '''
    trNumber = _trainNumber
    trSize = _trainSize
    teNumber = _testNumber
    return trNumber, trSize, teNumber

def classify(target):
    '''
    Using train and label set to check the classifier's error ration
    '''
    import datetime
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool()
    starttime = datetime.datetime.now()
    res = pool.map(kNNClassify, target)
    endtime = datetime.datetime.now()
    timeCost = (endtime - starttime).seconds
    print timeCost
    return res

def getErrorRate(trueLabel):
    '''
    Get the error rate of train samples
    '''
    errorRate = calcError(result, trueLabel)
    print errorRate

if __name__ == '__main__':
    #-------init option---------------------
    trainNumber, trainSize, testNumber = initOption(500, 500, 300)
    #-------init data-----------------------
    label, train, test = getDataArr()
    trainSample, labelSample, trainSub, labelSub, testSample = \
    getSubSample(trainNumber, trainSize, testNumber)
    #-------result--------------------------
    result = []
    #-------classify sub--------------------
    result = classify(trainSample)
    getErrorRate(labelSample)
    result = classify(testSample)
    saveResult(result, 'resultSub')
    #-------classify whole------------------
    trainNumber, trainSize, testNumber = initOption(0, len(train), len(test))
    trainSample, labelSample, trainSub, labelSub, testSample = \
    getSubSample(trainNumber, trainSize, testNumber)
    #-------wait----------------------------
    result = classify(testSample)
    saveResult(result, 'result')
    #-------end-----------------------------
    
