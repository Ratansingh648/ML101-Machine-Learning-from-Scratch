import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Loading the dataset
train = pd.read_csv("C://Users//Ratan Singh//Downloads//compressed//mnist_train.csv")
test = pd.read_csv("C://Users//Ratan Singh//Downloads//compressed//mnist_test.csv")

# Splitting dataset in features and target
X_train = train.iloc[:,0:784]
Y_train = train.iloc[:,784].ravel()

X_test = test.iloc[:,0:784]
Y_test = test.iloc[:,784].ravel()

# Defining model
model = Sequential()
model.add(Dense(512,input_dim = 784, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x = X_train,y=to_categorical(Y_train),epochs = 10)
