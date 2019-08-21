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
    

    return J