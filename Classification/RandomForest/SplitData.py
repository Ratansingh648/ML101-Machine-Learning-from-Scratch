import numpy as np


def splitData(X, y, columnIndex, threshold):

	X = np.array(X)
	y = np.array(y)

	lowerX = X[X[:,columnIndex] <= threshold,:]
	upperX = X[X[:,columnIndex] > threshold,:]
	
	lowerY = y[X[:,columnIndex] <= threshold]
	upperY = y[X[:,columnIndex] > threshold]

	return lowerX, upperX, lowerY, upperY		
