# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:25:40 2018

@author: topsec
"""

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt', 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    print(dataMatrix.transpose())
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    print('m:%d n:%d' % (m, n))
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat - h
        #print(error)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    
    dataMat, labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * array(dataMatrix[i])
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    trainingSet = []
    trainingLabels = []
    
    with open('horseColicTraining.txt', 'r') as frTrain:
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr =[]
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
            
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    print(trainWeights)
    errorCount = 0; numTestVec = 0.0
    with open('horseColicTest.txt', 'r') as frTest:
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr =[]
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
                errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__=='__main__':
    #dataMat, labelMat = loadDataSet()
    #weights = gradAscent(dataMat, labelMat)
    #weights = stocGradAscent0(dataMat, labelMat)
    #weights = stocGradAscent1(dataMat, labelMat)
    #plotBestFit(weights.getA())
    #plotBestFit(weights)
    multiTest()
    
    