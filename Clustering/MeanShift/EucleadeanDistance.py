import numpy as np

def eucleadeanDistance(x1,x2):
    x1 = np.array(x1)
    x2 = np.array(x2)

    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist
