import numpy as np

from ComputeNeighborhood import computeNeighborhood
from GaussianKernel import gaussianKernel
from RectangularKernel import rectangularKernel
from EpanechnikovKernel import epanechnikovKernel

def updateCentroid(X, centroid, kernel = 'epanechnikov', lowerLimit = 0, upperLimit = 10, spread = 1, neighborhood = 5):

    neighbors = computeNeighborhood(X,centroid,neighborhood)
    dist = np.sum((neighbors - centroid)**2,axis = 1)

    if kernel.lower() == "gaussian":
        weights = gaussianKernel(dist,spread)
    elif kernel.lower() == "rectangular":
        weights = rectangularKernel(dist, lowerLimit, upperLimit)
    elif kernel.lower() == 'epanechnikov':
        weights = epanechnikovKernel(dist)
    else:
        print("Warning: Kernel is not suppported. Using Gaussian Kernel instead")
        weights = gaussianKernel(dist,spread)
        
    normalWeights = weights.reshape([1,-1])/np.sum(weights)
    
    weightedNeighbors = np.matmul(normalWeights,neighbors).ravel()

    return weightedNeighbors
    
    
