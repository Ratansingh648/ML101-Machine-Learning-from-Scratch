import numpy as np
import pandas as pd

from ComputeNearestCluster import computeNearestCluster
from UpdateCentroid import updateCentroid
from InitCentroids import initCentroids


dataset = pd.read_excel('C:\\Users\\Ratan Singh\\Desktop\\ML Training Code\\KMeans\\iris.xlsx')
dataset = np.array(dataset)
n = dataset.shape[1] - 1
X = dataset[:,0:n]
y = dataset[:,n]


# Defining Clustering Parameters
K = 3
centroids = initCentroids(X,K)
num_iter = 9


# Clustering
for i in range(0,num_iter):
    clusters = computeNearestCluster(X,centroids)
    centroids = updateCentroid(X,clusters,K,'median')


clusters = np.squeeze(clusters)
print('Kmeans Clusering is Done. Following are the labels:')
print(clusters[0:10])
