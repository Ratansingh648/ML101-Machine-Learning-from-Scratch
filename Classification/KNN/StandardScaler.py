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
    return X, mu, sigma




def denormalize(X,mu,sigma):
	"""
	Write a code to perform the reverse operation of the above matrix

	X 		:  Normalized Feature matrix (m,n)
	mu  	:  mean array (n,)
	sigma	:  Standard deviation array (n,)

	Returns denormalized matrix X (m,n)
	"""    

	return X
