import numpy as np
from EucleadeanDistance import eucleadeanDistance

def checkNeighborhood(x,y,epsilon):
    """
    Return true if the distance between points x and y are lesser than epsilon
    """

    #Enter your code here
    
    d = eucleadeanDistance(x,y)
    return d < epsilon
