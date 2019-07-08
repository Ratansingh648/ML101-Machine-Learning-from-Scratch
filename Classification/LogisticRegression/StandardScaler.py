import numpy as np

def normalize(X):

    """
    Write a code to normalize the dataset as per the gaussian kernel

    X 		:  Denormalized Feature matrix (m,n)

    Returns 
    X 		: Normalized matrix (m,n)
    mu  	:  mean array (n,)
    sigma	:  Standard deviation array (n,)

    X_normalized = X - mu / sigma

    """


    X = np.array(X)
    
    mu = np.mean(X,axis = 0)
    sigma = np.std(X,axis = 0)
    X = (X-mu)/sigma
    return X, mu, sigma




def denormalize(X,mu,sigma):

    """
    Write a code to perform the reverse operation of the above matrix

    X 		:  Normalized Feature matrix (m,n)
    mu  	:  mean array (n,)
    sigma	:  Standard deviation array (n,)

    Returns denormalized matrix X (m,n)
    """
    
    X = np.array(X)
    X = X*sigma+mu
    return X
