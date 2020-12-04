import numpy as np


def gaussianKernel(X):

	X = np.array(X)

	mean = np.mean(X)
	std = np.std(X)
	z = np.divide((X-mean), std)
	scaling = 1/np.sqrt(2*np.pi)*sigma

	prob = scaling*np.exp(-0.5*z**2)

    return prob
