import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from ComputeDistance import computeDistance
from ComputeNeighbors import computeNeighbors
from StandardScaler import normalize

# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//Classification//KNN//PhishingWebsite.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

# Normalizing the data
X,mu,sigma = normalize(X)

# Splititng data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2)


# Defining parameters for KNN
K = 11
yp = []


# Predicting a KNN
for i in range(X_test.shape[0]):
	d = computeDistance(X_test[i,:],X_train,'manhattan')
	yi = computeNeighbors(d,Y_train,K)
	yp.append(yi)
	

# Prediction using the model
yp = np.array(yp)
print(confusion_matrix(yp,Y_test))

