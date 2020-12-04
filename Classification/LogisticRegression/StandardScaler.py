import numpy as np


def normalize(X):

    X = np.array(X)
    
    mu = np.mean(X,axis = 0)
    sigma = np.std(X,axis = 0)
    X = (X-mu)/sigma
    return X, mu, sigma




def denormalize(X,mu,sigma):

    X = np.array(X)
    X = X*sigma+mu
    return X
