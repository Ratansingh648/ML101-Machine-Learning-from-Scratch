import numpy as np


def gradientDescent(weight, gradient, alpha=0.1):
    """

    Gradient descent algorithm can be given as
    weight_new = weigh_old - learning_rate*gradient

    """	
    # Enter your code here
    weight = weight - alpha*gradient
    return weight

