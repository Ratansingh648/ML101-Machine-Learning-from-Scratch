import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ComputePolynomialFeatures import computePolynomialFeatures
from StandardScaler import normalize
from ComputeL2Cost import computeL2Cost
from ComputeGradient import computeGradient
from GradientDescent import gradientDescent
from R2 import R2


# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//Regression//PolynomialRegression//winequality-red.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

# Converting the features into Polynomial Features
X = computePolynomialFeatures(X,4)

# Normalizing the dataset
X, mu, sigma = normalize(X)

X = np.hstack((X, np.ones((X.shape[0],1))))

# Training the data into train and test 
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3)


# Defining parameters for gradient descent

initalWeights = np.random.random([1,X.shape[1]])
maxIter = 1000
learningRate = 0.00000005


# Training a Polynomial Regression
weights = initalWeights
cost = []

for i in range(maxIter):

	yp = np.matmul(X_train,weights.T).ravel()
	J = computeL2Cost(Y_train, yp)
	G = computeGradient(X_train, Y_train, yp)
	weights = gradientDescent(weights, G, learningRate)
        
	if i%10 ==0:
		print("Cost of the model is {}".format(J))
	cost.append(J)

print("Weights after the training are : {}".format(weights))


# Plotting the training loss curve
plt.plot(range(0,len(cost)),cost)
plt.title('Cost per iterations')
plt.show()


# Prediction using the model
yp = np.matmul(X_test,weights.T).ravel()
print("MSE for the fitted model is {}".format(computeL2Cost(Y_test,yp)))

R2_score = R2(Y_test,yp)
print("The variance explained by the model is {}".format(R2_score))
