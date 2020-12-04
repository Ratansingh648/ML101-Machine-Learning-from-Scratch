import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ComputeL2Cost import computeL2Cost
from ComputeDistance import computeDistance
from ComputeNeighbors import computeNeighbors
from StandardScaler import normalize
from R2 import R2


# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//Regression//KNN//BostonHousing.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

# Normalizing the data
X,mu,sigma = normalize(X)

# Splititng data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2)


# Defining parameters for KNN
K = 2
yp = []


# Predicting a KNN
for i in range(X_test.shape[0]):
	d = computeDistance(X_test[i,:],X_train,'manhattan')
	yi = computeNeighbors(d,Y_train,K)
	yp.append(yi)
	

# Prediction using the model
print("MSE for the fitted model is {}".format(computeL2Cost(Y_test,yp)))

R2_score = R2(Y_test,yp)
print("The variance explained by the model is {}".format(R2_score))

