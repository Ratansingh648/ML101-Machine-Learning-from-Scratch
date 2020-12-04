import numpy as np

# code to compute L2 Cost

def computeL2Cost(y,yp):

    # Write your code here
    y = np.array(y)
    yp = np.array(yp)
    m = len(y)    
    J = np.sum(np.power((y-yp),2))/(2*m)
    return J
