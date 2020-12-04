import numpy as np

from CheckNeighborhood import checkNeighborhood

def computeNeighborhood(X,pointIndex,epsilon):
    """
    Return a list of index of neighboring points which have
    distance less than epsilon than the given point
    """

    # Enter your code here
    
    neighbors = np.array([i for i in range(X.shape[0]) if checkNeighborhood(X[i],X[pointIndex],epsilon)])
    return neighbors
