import numpy as np

def computeGradient(X,y,yp):

    # Enter your code here
    X = np.array(X)
    y = np.array(y)
    yp = np.array(yp)
    m = X.shape[0]

    gradient = np.matmul((yp-y).reshape([-1,1]).T,X)/m
    return gradient
    

    
    
    
