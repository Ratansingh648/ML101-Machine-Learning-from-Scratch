import numpy as np


def computeSplits(X):

	attributeSplits = {}
	for attributeIndex in range(0,X.shape[1]):
		uniquePoints = np.unique(X[:,attributeIndex])
		splits = [(uniquePoints[i] + uniquePoints[i+1])/2.0 for i in range(len(uniquePoints)-1)]
		attributeSplits[attributeIndex] = splits

	return attributeSplits		
