import numpy as np

from ComputeNeighborhood import computeNeighborhood


def initClustering(X, y, pointIndex, clusterIndex, epsilon, minPoints):
    """
    Return true if the distance between points x and y are lesser than epsilon
    """

    #Enter your code here
    
    neighborhood = computeNeighborhood(X, pointIndex, epsilon)


    # Check if it is not a noise
    if len(neighborhood) < minPoints:
    	y[pointIndex] = 0

    else:
    	y[neighborhood] = clusterIndex


    # Iterate over the neighbors to find the next neighborhood
    for i in neighborhood:
    	new_neighborhood = computeNeighborhood(X,i,epsilon)
    	for i in new_neighborhood:
