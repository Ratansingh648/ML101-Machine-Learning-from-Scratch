import numpy as np


def adam(weight, gradient, velocity, rms ,alpha=0.1,beta1 = 0.9, beta2=0.99):
	
    # Enter your code here
    velocity = beta1*velocity + (1-beta1)*gradient
    rms = beta2*rms + (1-beta2)*gradient 
    weight = weight - alpha*np.divide(velocity,rms)
    return weight

