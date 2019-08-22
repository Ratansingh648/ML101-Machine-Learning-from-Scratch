import numpy as np

def computeEntropy(y):
	_ , counts = np.unique(y, return_counts = True)
	prob = counts / np.sum(counts)
	logProb = np.log2(prob)
	entropy = -1*np.sum(np.multiply(prob,logProb))
	return entropy
    
