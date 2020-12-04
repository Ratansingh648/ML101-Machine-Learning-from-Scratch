import numpy as np

# code to compute L2 Cost

def computeEntropyCost(y,yp):

    # Write your code here
    y = np.array(y).ravel()
    yp = np.array(yp).ravel()
    m = len(y)    
    J = -1*np.sum(y*np.log(yp+0.0000001)+(1-y)*np.log(1+0.0000001-yp))/m
    return J
