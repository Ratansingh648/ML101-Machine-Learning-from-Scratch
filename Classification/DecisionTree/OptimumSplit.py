import numpy as np

from ComputeBranchEntropy import computeBranchEntropy
from SplitData import splitData

def optimumSplit(X, y, splits):

        X = np.array(X)
        y = np.array(y)

        optimumSplittings = {}
        minimumEntropy = 10000
        optimumColumn = None
        optimumThreshold = None
        
        for columnIndex, thresholdList in splits.items():
                for threshold in thresholdList:
                        x1,x2,y1,y2 = splitData(X, y, columnIndex, threshold)
                        if computeBranchEntropy(y1,y2) < minimumEntropy:
                                minimumEntropy = computeBranchEntropy(y1,y2)
                                optimumColumn = columnIndex
                                optimumThreshold = threshold

        optimumSplittings[optimumColumn] = optimumThreshold
        return optimumSplittings
    
