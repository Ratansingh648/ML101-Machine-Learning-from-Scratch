import numpy as np

# code to compute Nearest Neightbors
"""
K Neighbors with lowest distance are returned. 
Returns the class of majority votes
Returns a array (1,)

Note: As a practice, also write a code handling case when no class has a majority

"""

def computeNeighbors(d,y,K):
    # Write your code here
    return max_class