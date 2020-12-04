import numpy as np

# code to compute Hinge Loss
"""
Hinge Loss is specifically concerned with misclassification only

J = mean(1-y(WX))

J : Cost
X : Feature matrix  (m,n)
W : Weight matrix   (1,n)
y : target matrix   (m,)

Returns a scaler

"""

def computeHingeCost(X,y,weight):

    # Write your code here
    y = np.array(y).reshape([-1,1])
    weight = np.array(weight).reshape([1,-1])

    m = y.shape[0]    

    X = np.array(X).reshape([m,-1])
    J = np.sum(np.maximum(1-(y.ravel()*np.matmul(X,weight.T).ravel()),0))/m
    return J