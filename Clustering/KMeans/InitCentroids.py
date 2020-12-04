import numpy as np
import random

def initCentroids(X,K):

	# Enter your code here
    m,n = X.shape
    max_list = [max(X[:,x]) for x in range(0,n)]
    min_list = [min(X[:,x]) for x in range(0,n)]
    centroids = np.zeros((K,n)) 
    
    for i in range(0,K):
        centroids[i,:] = np.array([random.uniform(min_list[x],max_list[x]) for x in range(0,len(max_list))]) 
    return centroids
