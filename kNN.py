'''
Machine Learning in Action
codes with Python3
'''

# from numpy import *
import numpy as np
import operator

def createDataSet(): 
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k): 
    dataSetSize = dataSet.shape[0]
    # np.tile(A, reps)结果的维度是max(A.ndim, reps.length)其中reps类型为元组，较短的用1在前面补齐
    # 以二维为例np.tile([a, b], (2, 3))结果是array([[a, b, a, b, a, b], [a, b, a, b, a, b]])
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # print(diffMat)
    sqDiffMat = diffMat**2
    # print(sqDiffMat)
    sqDistance = sqDiffMat.sum(axis=1)
    # print(sqDistance)
    distances = sqDistance**0.5
    # print(distances)
    # np.argsort()返回原数组中值从小到大的索引
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)
    classCount = {}
    for i in range(k): 
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(voteIlabel)
        # dict.get(key, default=None)取出key的值，如果没有值设置默认值为default
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print(classCount)
    # dict.items()方法：python3中返回一个迭代器，对应python2 中的dict.iteritems()方法
    # sorted(iterable, cmp=None, key=None, reverse=False)
    # 产生一个新列表，需要原列表是可迭代的，cmp是用于比较的函数，key是比较关键字，cmp和key可以使用lambda函数，reverse=True逆序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount)
    # classCount = {'B': 2, 'A': 1}
    # sortedClassCount = [('B', 2), ('A', 1)]
    return sortedClassCount[0][0]
