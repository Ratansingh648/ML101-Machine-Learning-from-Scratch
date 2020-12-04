import numpy as np

# code to compute Nearest Neightbors
"""
K Neighbors with lowest distance are returned

Returns a array (K,)

"""

def computeNeighbors(d,y,K):
    # Write your code here
    y = np.array(y).ravel()
    d = np.array(d).ravel()

    d,y = zip(*sorted(zip(d,y)))

    y = np.mean(y[0:K])
    return y