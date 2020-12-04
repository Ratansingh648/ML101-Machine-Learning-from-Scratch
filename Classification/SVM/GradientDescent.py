import numpy as np


def gradientDescent(weight, gradient, alpha=0.1):
	
    # Enter your code here
    weight = weight - alpha*gradient
    return weight

