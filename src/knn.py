import csv
import pandas
import math
import numpy as np


def readFile(dataFile, columns):
    f = open(dataFile, 'r')
    dataList = []
    for x in f:
        x = x.strip('\n')
        ar = x.split(' ')
        dataList.append(ar)
    df = pandas.DataFrame(np.asarray(dataList), index=None, columns=columns)
    return df


def visualizeData(dataFile, columns):
    print(readFile(dataFile, columns))


def minkowski(a, b, r=1):
    sum = 0
    for i in range(len(a)):
        sum = sum + (abs(a[i] - b[i])**(1/r))
    return sum


def DistTrainDataTestData(trainFile, testFile, columns, dataColumns):
    train = readFile(trainFile, columns)
    trainNumeric = train[dataColumns].apply(pandas.to_numeric)
    trainList = trainNumeric.values.tolist()
    test = readFile(testFile, columns)
    testNumeric = test[dataColumns].apply(pandas.to_numeric)
    testList = testNumeric.values.tolist()
    for testIndex in range(len(testList)):
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
    return max(list, key=list.count)


def setClassToTest(testFrame, listClasses):
    testFrame['class predicted'] = pandas.Series(listClasses, index=None)
    #print(testFrame)
    return testFrame

def precision(datatest):
    columnClasses = datatest[['class','class predicted']]
    numericColumnClasses = columnClasses.apply(pandas.to_numeric)
    listColumnClasses = numericColumnClasses.values.tolist()
    TP = 0
    for i in range(len(listColumnClasses)):
        #print(listColumnClasses[i][0], listColumnClasses[i][1])
        if listColumnClasses[i][0] == listColumnClasses[i][1]:
            TP = TP+1
    #print(TP)
    #print(len(listColumnClasses) - TP)
    return TP/ len(listColumnClasses)

def searchBestK(train, test ):
    i = 0
    prec = 0
    maxPrecision = 0
    k = 2 * i + 1
    while k < 50:
        classes = knnClaissifyer(train, k)
        prec = precision(setClassToTest(test, classes))
        if prec > maxPrecision:
            testclass = setClassToTest(test, classes)
            maxPrecision = prec
            bestK = k
        elif prec == 100:
            print(setClassToTest(test, classes))
            return k, prec
        i = i + 1
        k = 2 * i + 1
    print('\n')
    print(testclass)
    print('\n')
    return bestK, maxPrecision

visualizeData("../data/iris/iris.tst", ['x1', 'x2', 'x3', 'x4', 'class'])
#DistTrainDataTestData("../data/iris/iris.trn", "../data/iris/iris.tst")
train, test = DistTrainDataTestData("../data/iris/iris.trn", "../data/iris/iris.tst", ['x1', 'x2', 'x3', 'x4', 'class'], ['x1', 'x2', 'x3', 'x4'])
k, p = searchBestK(train, test)
print('\n')
print("Meilleur k :", k, "Meilleur precesion :", p)
#classes = knnClaissifyer(train,5)
#print("******************")
#print(classes)
#print("******************")
#setClassToTest(test, classes)
#print(precision(setClassToTest(test, classes)))

#visualizeData("../data/iris/iris.tst", ['x1', 'x2', 'x3', 'x4', 'class'])
#print(len(readFile("../data/iris/iris.trn")))

exit(0)






