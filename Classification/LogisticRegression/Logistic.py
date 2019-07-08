import numpy as np 


def logistic(weights, X):

	"""
	Logistic function maps real valued functions to range of 0 to 1
	logistic(z)  = 1 /  [1 + exp(-z) ]
	where z = WX
	
	W  : Weights
	X  : Features

	Returns (m,1) array

	"""
	# Enter your code here


	weights = np.array(weights)
	X = np.array(X)
	z = np.matmul(X,weights.T)

	deno = 1 + np.exp(-1*z)
	return 1/deno