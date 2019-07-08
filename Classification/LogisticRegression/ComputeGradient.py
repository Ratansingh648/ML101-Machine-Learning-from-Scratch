import numpy as np

def computeGradient(X,y,yp):

    """
    Gradient can be computed by chain rule of differentiation.
    on differentiating the entropy cost function with respect to w
    J = - mean[ylog(yp)+(1-y)log(1-yp)]
    we get

    G = mean[(yp(1-y)-y(1-yp))X]
    Dimension of gradient should be same as the dimension of weight matrix

    J  :  Cost funtion
    y  :  True target value
    yp :  Predicted target value
    X  :  Features

    """

    # Enter your code here
    X = np.array(X)
    y = np.array(y).reshape([-1,1])
    yp = np.array(yp).reshape([-1,1])
    
    m = y.shape[0]
    
    gradient = np.matmul((yp*(1-y)-y*(1-yp)).T,X)/m
    
    return gradient
    

    
    
    
