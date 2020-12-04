import numpy as np

def computeGradient(X,y,yp):

	# Enter your code here
    X = np.array(X)
    y = np.array(y).reshape([1,-1])
    yp = np.array(yp).reshape([1,-1])
    m = y.shape[1]

    gradient = np.sum(np.matmul((yp-y),X))/m
    return gradient
    

    
    
    
