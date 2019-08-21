import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from ComputeHingeCost import computeHingeCost
from ComputeGradient import computeGradient
from GradientDescent import gradientDescent
from StandardScaler import normalize

# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//Classification//SVM//Banknote_authentication.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

# Converting on scale of (1,-1)
y = 2*y-1

# Notmalizing the dataset
X,mu,sigma = normalize(X)

# Adding ones for bias unit
X = np.hstack((X, np.ones((X.shape[0],1))))


# Splitting data into train test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2)


# Defining parameters for gradient descent
initalWeights = np.random.random([1,X.shape[1]])
maxIter = 100
learningRate = 0.1


# Training a SVM 
weights = initalWeights
cost = []

for i in range(maxIter):

	J = computeHingeCost(X_train,Y_train,weights)
	G = computeGradient(X_train, Y_train, weights)
	weights = gradientDescent(weights, G, learningRate)

	if i%10 ==0:
		print("Cost of the model is {}".format(J))
	cost.append(J)


print("Weights {} after the training are".format(weights))


# Plotting the training loss curve
plt.plot(range(0,len(cost)),cost)
plt.title('Cost per iterations')
plt.show()


# Prediction using the model
yp = (np.matmul(X_test,weights.T).ravel() >= 1)*2-1
print(confusion_matrix(yp,Y_test))

