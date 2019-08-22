import numpy as np

from ComputeEntropy import computeEntropy

def computeBranchEntropy(y1,y2):
	prob1  = len(y1) / (len(y1) + len(y2))
	prob2 = 1 - prob1

	entropy1 = computeEntropy(y1) 
	entropy2 = computeEntropy(y2)

	branchEntropy = prob1*entropy1 + prob2*entropy2

	return branchEntropy
    
