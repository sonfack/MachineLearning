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
    test = readFile(testFile)
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
    #print(train)
    return train, test

def knnClaissifyer(distDataFrame, k=1):
    classes = []
    for j in range(50):
        listtrie = distDataFrame.sort_values(by=['dist_t'+str(j+1)])
        #print(listtrie)
        list  = []
        for i in range(k):
            list.append(listtrie.iloc[i][4])
        classes.append(voteClass(list))
    #print(classes)
    return classes


def voteClass(list):
    print(list)
    val0=0
    val1=0
    val2=0
    for i in range(len(list)):
        if int(list[i])==0:
           val0 = val0 + 1
        elif int(list[i])==1:
           val1 = val1 + 1
        elif int(list[i])==2:
           val2 = val2 + 1
    maxi = 0

    maxi = max(float(val0), float(val1), float(val2))
    if maxi == val0:
        return 0
    elif maxi == val1:
        return 1
    else:
        return 2


def setClassToTest(testFrame, listClasses):
    testFrame['class predicted'] = pandas.Series(listClasses, index=None)
    print(testFrame)


#DistTrainDataTestData("../data/iris/iris.trn", "../data/iris/iris.tst")
train, test = DistTrainDataTestData("../data/iris/iris.trn", "../data/iris/iris.tst")
classes = knnClaissifyer(train,3)
print("******************")
print(classes)
print("******************")
setClassToTest(test, classes)

#visualizeData("../data/iris/iris.tst")
#print(len(readFile("../data/iris/iris.trn")))

exit(0)






