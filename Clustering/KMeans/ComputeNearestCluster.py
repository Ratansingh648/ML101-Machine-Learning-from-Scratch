import numpy as np

def computeNearestCluster(X,centroids):
    m = X.shape[0]              # number of examples
    K = centroids.shape[0]      # number of clusters
    clusters = np.zeros(m)
    
    for i in range(0,m):
        min_dist = 100000       # A very large number
        
        for j in range(0,K):
            dist = np.squeeze(np.sqrt(np.sum(np.square(np.array(X[i,:])-np.array(centroids[j,:])))))
            
            if dist < min_dist:
                min_dist = dist
                clusters[i] = j
    return clusters
