import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from ComputeEntropy import computeEntropy
from ComputeSplits import computeSplits
from OptimumSplit import optimumSplit
from SelectClass import selectClass
from SplitData import splitData

# Reading and loading X and y
dataset = pd.read_csv("Banknote_authentication.csv")
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])


# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.2)


# Random Forest parameters
minSampleSplit = 20
maxDepth = None
numberEstimators = 101
subSample = 0.8
bootStrap = False

# Recursive method to generate a tree
def createTree(X, y, minSampleSplit = 2, maxDepth = None, depth = 0):

        X = np.array(X)
        y = np.array(y)

        tree = {} 
        depth = depth + 1

        if ((isinstance(maxDepth, (int,float)) and depth <= maxDepth) or maxDepth==None) and len(y) >= minSampleSplit and computeEntropy(y) != 0:
                splits = computeSplits(X)
                optimumSplitList = optimumSplit(X, y, splits)

                columnIndex = list(optimumSplitList.keys())[0]
                threshold = list(optimumSplitList.values())[0]

                lowerX, upperX, lowerY, upperY = splitData(X, y, columnIndex, threshold)
                lowerNode = createTree(lowerX, lowerY, minSampleSplit = minSampleSplit, depth = depth, maxDepth = maxDepth)
                upperNode = createTree(upperX, upperY, minSampleSplit = minSampleSplit, depth = depth, maxDepth = maxDepth)

                tree[columnIndex] = [threshold, lowerNode, upperNode]
        else:
                return selectClass(y)
        return tree




# Training a Random Forest
trees = []

for i in range(numberEstimators):
	subSampleSize = round(subSample*X_train.shape[0])

	# Random sampling to create subsets of observations and features
	order = np.random.choice(range(X_train.shape[0]),size = subSampleSize, replace = bootStrap)
	X_train_sample = X_train[order,:]
	Y_train_sample = Y_train[order]
	
	tree = createTree(X_train_sample, Y_train_sample, minSampleSplit = minSampleSplit, maxDepth = None)
	trees.append(tree)




# Predicting the output
def predictTree(x, tree):
        columnIndex = list(tree.keys())[0]
        threshold, left, right = tree[columnIndex]

        answer = left if x[columnIndex] <= threshold else right

        if isinstance(answer,dict):
                return predictTree(x, answer)
        else:
                return answer

yp = []
for i in range(X_test.shape[0]):
	outputClass = selectClass([predictTree(X_test[i,:], tree) for tree in trees])
	yp.append(outputClass)

yp = np.array(yp)
# Printing the confusion matrix
print(confusion_matrix(Y_test,yp))
