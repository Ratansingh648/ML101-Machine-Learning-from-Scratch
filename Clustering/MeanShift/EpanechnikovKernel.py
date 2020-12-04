import numpy as np

def epanechnikovKernel(dist):

    limitBound = (dist <= 1)*1.0
    kernelMap = 0.75*(1-dist**2)

    prob = np.multiply(limitBound,kernelMap)

    return prob    
