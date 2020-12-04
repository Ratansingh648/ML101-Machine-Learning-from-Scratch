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

    y = list(y[0:K])
    max_class = max(set(y),key=y.count)
    
    if len(list(set(y))) != len(y) or len(y)==1:
    	return max_class 
    else:
    	print("Warning : Conflict has arised. Please increase Neighbors.")
    	return None
