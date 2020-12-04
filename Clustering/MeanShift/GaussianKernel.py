import numpy as np

def gaussianKernel(dist,spread = 1):

    dist = np.array(dist)
    
    deno = 1.0/(np.sqrt(2*np.pi)*spread)
    num = np.exp(-0.5*(dist/spread)**2)
    prob = num/deno

    return prob
