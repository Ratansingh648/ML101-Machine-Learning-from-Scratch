import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from Logistic import logistic
from ComputeEntropyCost import computeEntropyCost
from ComputeGradient import computeGradient
from GradientDescent import gradientDescent
from StandardScaler import normalize, denormalize

# Reading and loading X and y
dataset = pd.read_csv("C://Users//Ratan Singh//Desktop//ML Training Code//Classification//LogisticRegression//BreastCancer.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

# Normalizing the features

X,mu,sigma = normalize(X)


# Adding ones for bias
X = np.hstack((X, np.ones((X.shape[0],1))))


# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2)


# Defining parameters for gradient descent
initalWeights = np.random.random([1,X.shape[1]])
maxIter = 1000
learningRate = 0.3


# Training a Logistic Regression
weights = initalWeights
cost = []

for i in range(maxIter):

	yp = logistic(weights, X_train)
	J = computeEntropyCost(Y_train, yp)
	G = computeGradient(X_train, Y_train, yp)
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
yp = logistic(weights, X_test)
yp = ((yp >= 0.5)*1.0).ravel()

# Evaluation using confusion matrix
print(confusion_matrix(yp,Y_test))

