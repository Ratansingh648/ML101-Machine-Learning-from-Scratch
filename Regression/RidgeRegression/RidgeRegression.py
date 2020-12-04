import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

from ComputeRidgeCost import computeRidgeCost
from ComputeGradient import computeGradient
from GradientDescent import gradientDescent

# Reading and loading X and y
dataset = pd.read_excel("C://Users//Ratan Singh//Desktop//ML Training Code//LinearRegression//cancer_reg.xlsx")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:4])
y = np.array(dataset.iloc[:,4])

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3)


# Defining parameters for gradient descent

initalWeights = np.random.random([1,X.shape[1]])
maxIter = 100
learningRate = 0.1


# Training a Logistic Regression

weights = initalWeights
cost = []

for i in range(maxIter):

	yp = np.matmul(weights, X_train.T).ravel()
	J = computeL2Cost(Y_train, yp)
	G = computeGradient(X_train, Y_train, yp)
	weights = gradientDescent(weights, G, learningRate)

	if i%10 ==0:
		print("Cost of the model is {}".format(J))
	cost.append(J)


print("Weights {} after the training are".format(weights))



# Prediction using the model

yp = np.matmul(weights, X_test.T).ravel()
print("MSE for the fitted model is {}".format(computeL2Cost(Y_test,yp)))


