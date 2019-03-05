import csv
import pandas

import numpy as np


def readFile(datafile):
    f = open(datafile, 'r')
    dataList = []
    for x in f:
        x = x.strip('\n')
        ar = x.split(' ')
        dataList.append(ar)
    df = pandas.DataFrame(np.asarray(dataList), index=None, columns=['x1', 'x2', 'x3', 'x4', 'class'])
    return df


def visualizeData(datafile):
    print(readFile(datafile))

def minkowski(a, b):
    sum = 0
    for i in range(len(a)):
        sum = sum + abs(a[i] - b[i])
    return sum


def DistTrainDataTestData(trainFile, testFile):
    train = readFile(trainFile)
    trainNumeric = train[['x1','x2','x3','x4']].apply(pandas.to_numeric)
    trainList = trainNumeric.values.tolist()
    test  = readFile(testFile)
    testNumeric = test[['x1','x2','x3','x4']].apply(pandas.to_numeric)
    testList = testNumeric.values.tolist()
    for testIndex in range(len(testList)):
        print('\n')
        print(testList[testIndex])
        print('\n')
        listDistance = []
        for trainIndex in range(len(trainList)):
            #print(trainList[trainIndex])
            listDistance.append(minkowski(testList[testIndex], trainList[trainIndex]))
        train['dist_t'+str(testIndex+1)] = pandas.Series(listDistance, index=None)
    print(train)



DistTrainDataTestData("../data/iris/iris.trn", "../data/iris/iris.tst")
#visualizeData("../data/iris/iris.tst")
#print(len(readFile("../data/iris/iris.trn")))

exit(0)






