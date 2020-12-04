import numpy as np

def rectangularKernel(dist, lowerLimit, upperLimit):

    lowerBound = dist >= lowerLimit
    upperBound = dist <= upperLimit

    bounded = lowerBound + upperBound

    return bounded*1.0
