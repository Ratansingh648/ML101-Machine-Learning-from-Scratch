import numpy as np

"""
On differentiating the Hinge Loss with respect to w, we get gradient

G 	= -mean(Xy) 	for y(WX) < 0
	= 0				for y(WX) >= 0


G : Gradient
X : Feature matrix  (m,n)
W : Weight matrix   (1,n)
y : target matrix   (m,)

Returns a (1,n) array 
 
"""

def computeGradient(X,y,weight):

	# Enter your code here
    X = np.array(X)
    y = np.array(y).reshape([-1,1])
    m = X.shape[1]

    posCost = (np.maximum(1-(y.ravel()*np.matmul(X,weight.T).ravel()),0) > 0)*1.0 
    y = (posCost*y.ravel()).reshape([-1,1])

    gradient = -1*(np.matmul(y.T,X))/m
    return gradient
    
    
    
