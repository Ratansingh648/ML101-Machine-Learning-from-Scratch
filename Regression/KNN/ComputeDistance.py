import numpy as np


"""
Computes the following distance

eucleadean = sqrt(sum(|x-y|^2))
manhattan = (sum(|x-y|))
cosine = dot(X,Y)/(mag(X)*mag(Y))

x   :  First vector
y   :  second vector 

Returns a (m,1) array 
 
"""

def computeDistance(p,X, distanceType = "eucleadean"):
    # Enter your code here
    X = np.array(X)
    p = np.array(p).ravel()
    
    if distanceType.lower() == "eucleadean":
    	distance = np.sqrt(np.sum(np.square(X-p),axis=1))
    	
    elif distanceType.lower() == "manhattan":
    	distance = np.sum(np.abs(X-p),axis=1)
    	
    distance = np.array(distance)
    distance = distance.ravel()
    
    return distance
    
    
    
