import numpy as np


def selectClass(y):
    classes,counts = np.unique(y, return_counts = True)
    classChosen = classes[np.argmax(counts)]
    return classChosen 
