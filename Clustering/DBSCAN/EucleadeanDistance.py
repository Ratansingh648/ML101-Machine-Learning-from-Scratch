import numpy as np


def eucleadeanDistance(x,y):
    """
    Compute Eucleadean distance between two points
    D(x,y) = sqrt[sum[(x-y)^2]]

    Note: Different Distances will lead to different clusterings
    
    """

    x = np.array(x)
    y - np.array(y)
    D = np.sqrt(np.sum(np.square(x-y)))

    return D
