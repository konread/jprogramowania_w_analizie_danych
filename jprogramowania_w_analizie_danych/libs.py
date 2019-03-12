import re as re
import numpy as np
from collections import Counter

def readByColumn(filename):
    content = [[]]
    with open(filename) as f:
        for x in [re.sub("\s+", ",", res.strip()) for res in f.readlines()]:
            if len(content) == 1:
                content = [[item] for item in x.split(',')]
                continue
            for colNum, value in enumerate(x.split(',')):
                content[colNum].append(value)
    return content

def toFloat(array):
    return [float(x) for x in array]

def findMedian(array):
    return np.median(toFloat(array))

def findMax(array):
    return np.max(toFloat(array))

def findMin(array):
    return np.min(toFloat(array))

def findDominants(array):
    founded = Counter(array)
    print(founded)
    maxValue=0
    result=[]
    for item in founded:
        newValue = founded[item]
        if newValue > maxValue:
            maxValue = newValue
            result = []
            result.append(item)
        elif (newValue == maxValue) and (newValue not in result):
            result.append(item)
    return result
