import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def computePolynomialFeatures(X, degree=2):

    # Enter your code here
    X = np.array(X)
    polyFeatures = PolynomialFeatures(degree)
    X_poly = polyFeatures.fit_transform(X)[:,1:]
    
    return X_poly
    

    
    
    
