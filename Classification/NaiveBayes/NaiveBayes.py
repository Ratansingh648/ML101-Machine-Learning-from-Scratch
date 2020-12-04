import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Reading and loading X and y
dataset = pd.read_csv("Banknote_authentication.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])


# Splitting in train test 
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2)


# Defining a function to determine if value is categorical or not
def isCategorical(feature, threshold = 20):
	uniqueValues = len(np.unique(feature))
	return True if uniqueValues <= threshold else False


# Defining parameters for gradient descent
classes, priors = np.unique(Y_train, return_counts = True)
classes = list(classes)
priors = priors / np.sum(priors) 
likelihood = {}

# Training a Naive Bayes - Learning Likelihood
for classInstance in classes:
	order = np.array(Y_train) == classInstance
	X_class = X_train[order,:]

	featureDict = {}

	for featureIndex in range(0,X_class.shape[1]):
		if isCategorical(X[:,featureIndex]):
			variable,count = np.unique(X[:,featureIndex], return_counts = True)
			count = count / len(X_class.shape[0])
			featureDict[featureIndex] = ["categorical",variable,count]
		else:
			mean = np.mean(X[:,featureIndex])
			std = np.std(X[:,featureIndex])
			featureDict[featureIndex] = ["continuous", mean, std]

	likelihood[classInstance] = featureDict


# Predicting the classes
for classInstance in classes:
        classLikelihood = priorAttributes[classInstance]
        X_temp = X_test
        
        for featureIndex, likelihoodProb in classLikelihood.items():
                if likelihoodProb[0] == "continuous":
                        mean = likelihoodProb[1]
                        std = likelihoodProb[2]
                        X_temp[:,featureIndex] = (X_temp[:,featureIndex] - mean)/std
                else:
                        featureType,priors = np.unique(X_temp[:,featureIndex], return_counts = True)
                        
                        





"""
# Plotting the training loss curve
plt.plot(range(0,len(cost)),cost)
plt.title('Cost per iterations')
plt.show()


# Prediction using the model
yp = (np.matmul(X_test,weights.T).ravel() >= 1)*2-1
print(confusion_matrix(yp,Y_test))
"""
