import numpy as np

# code to compute L2 Cost

def computeRidgeCost(y,w,regularizer = 0.1):

    # Write your code here
    y = np.array(y)
    yp = np.array(np.matmul(X,w.T))
    m = len(y)    
    J = (np.sum(np.power((y-yp),2)) + np.sum(regularizer*np.square(w)))/(2*m)
    return J
