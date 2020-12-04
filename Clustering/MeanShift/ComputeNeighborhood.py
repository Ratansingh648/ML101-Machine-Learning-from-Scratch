import numpy as np


def computeNeighborhood(X, centroid, threshold = 5):

    X = np.array(X)
    centroid = np.array(centroid)
    
    distanceMatrix = np.sum((X-centroid)**2,axis=1) < threshold
    neighboringPoints = X[distanceMatrix,:]

    return neighboringPoints
    
