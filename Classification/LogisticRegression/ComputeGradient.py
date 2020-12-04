import numpy as np

def computeGradient(X,y,yp):

    # Enter your code here
    X = np.array(X)
    y = np.array(y).reshape([-1,1])
    yp = np.array(yp).reshape([-1,1])
    
    m = y.shape[0]
    
    gradient = np.matmul((yp*(1-y)-y*(1-yp)).T,X)/m
    
    return gradient
    

    
    
    
