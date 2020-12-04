import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from UpdateCentroid import updateCentroid
from StandardScaler import normalize

# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//Clustering//MeanShift//iris.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

# Normalizing the data
X,mu,sigma = normalize(X)


# Defining parameters for Mean shift algorithm
distanceThreshold = 1
maxIter = 50
window = 0.5

# Defining placeholder for clustrering
centroids = np.zeros(X.shape)
temp = X


# Clustering using the mean shift algorithm
for i in range(maxIter):
    for j in range(X.shape[0]):
        centroids[j,:] = updateCentroid(X,temp[j,:],kernel = "gaussian",spread = window)
    temp = centroids


plt.scatter(X[:,2],X[:,3],color = 'blue')
plt.scatter(centroids[:,2],centroids[:,3],color='black')
plt.show()


