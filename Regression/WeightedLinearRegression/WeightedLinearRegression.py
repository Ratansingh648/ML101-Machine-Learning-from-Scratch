import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ComputeL2Cost import computeL2Cost
from ComputeGradient import computeGradient
from GradientDescent import gradientDescent
from StandardScaler import normalize
from R2 import R2

# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//Regression//LinearRegression//BostonHousing.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n]).ravel()

# Normalizing the dataset
X, mu, sigma = normalize(X)

X = np.hstack((X, np.ones((X.shape[0],1))))


# Splitting dataset in train test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3)


# Defining parameters for gradient descent
initalWeights = np.random.random([1,X.shape[1]])*0
maxIter = 1000
learningRate = 0.01


# Training a Least Squares Regression
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


# Computing the Residuals of the Training data
yp_train = np.matmul(X_train,weights.T).ravel()
residual = (yp_train - Y_train)


# Either the absolute value or the square of epsilon is fitted against the predictors.
# Weight after fitting represent the variance. Penalization to variance is stronger in Squared
squared_residuals = residual**2
absolute_residuals = abs(residual)
r = squared_residuals


#Fitting a OLS
residual_initialWeights = np.random.random([1,X_train.shape[1]])
residual_maxIter = 1000
learningRate = 0.1


# Training a OLS Regression on Residual vs Predictors
residual_weights = residual_initialWeights

for i in range(residual_maxIter):
	rp = np.matmul(X, residual_weights.T)
	G_residual = computeGradient(X, r, rp)
	residual_weights = gradientDescent(residual_weights, G_residual, learningRate)

print("Residuals Weights after the training are :: {}".format(residual_weights))


# Variances will be predicted residuals with OLS
variances = np.matmul(X_train, residual_weights.T)


# Weights are reciprocal of this or reciprocal of square root of this (Depends from sources to sources)
regression_weights = 1.0/np.sqrt(variances)
regression_weights_matrix = np.diagflat(np.array(regression_weights).ravel())


# Scaling the input data
X_train_weighted = np.dot(regression_weights_matrix, X_train)
Y_train_weighted = np.dot(regression_weights_matrix, Y_train.reshape([-1,1]))

X_test_weighted = np.dot(regression_weights_matrix, X_test)


# Learning the weights of Weighted Regression
WLS_initialWeights = np.random.random([1,X_train_weighted.shape[1]])
WLS_maxIter = 1000
learningRate = 0.1


# Training a OLS Regression on Residual vs Predictors
WLS_weights = WLS_initialWeights

for i in range(WLS_maxIter):
	yp_train_weighted = np.matmul(X_train_weighted, WLS_weights.T)
	G_weighted = computeGradient(X_train_weighted, Y_train_weighted, yp_train_weighted)
	WLS_weights = gradientDescent(WLS_weights, G_weighted, learningRate)

print("Weighted Regression Coefficients after the training are :: {}".format(WLS_weights))


# Prediction using the model
yp_weighted = np.matmul(X_test_weighted,WLS_weights.T).ravel()
variance_weights_matrix = np.diagflat(np.array(variances).ravel())
yp = np.dot(variance_weights_matrix, yp_weighted.reshape([-1,1]))


print("MSE for the fitted model is {}".format(computeL2Cost(Y_test,yp)))

R2_score = R2(Y_test,yp)
print("The variance explained by the model is {}".format(R2_score))
