import numpy as np

def updateCentroid(X,y,K,method = 'mean'):

    # Enter your code here

    centroids = np.zeros((K,X.shape[1])) 
    
    for i in range(0,K):
        index = np.where(y==i)[0]
        temp = X[index,:]
        
        if method == 'median':
            tempCentroid = np.median(temp,axis=0)
        elif method == 'mean':
            tempCentroid = np.mean(temp,axis=0)
    
        centroids[i,:] = np.array(tempCentroid)
    return centroids
