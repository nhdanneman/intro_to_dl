''' This code accompanies slides XX - XX in the Intro to Deep Learning with Keras course.
Author: Nathan Danneman
Created: 2020/01/18
Updated: 2020/01/18
'''

# This code walks through data preparation and model estimation for feed-forward neural networks.

# Assuming you've 'pip install keras' someplace this code can touch...

# imports
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from numpy import genfromtxt  # to read in the iris csv
import numpy as np
import pandas as pd
import collections

# where is the data?
raw_iris_data = pd.read_csv('/Users/ndanneman/Documents/datamachines/gits/intro_to_dl/data/iris1.csv', delimiter=',',dtype=None)
raw_iris_data.shape


# Let's glance at the raw data:
raw_iris_data.head(5)
# 4 continuous variables and a species designator

# Look at the DV:
collections.Counter(raw_iris_data["Species"])

# Build X and Y arrays
X = np.array(raw_iris_data.iloc[:,0:4])
# Easy eay to turn string categorical into numeric indicator, if you know the strings
Y = np.array(raw_iris_data.Species.map(dict(versicolor=1, virginica=0)))

## Make a simple train/test split -- note lack of stratification!
# Random sample of the training indices
train_indices = np.random.choice(range(len(Y)), 70, False)
# All indices
all_indices = np.array(range(100))
# Those in "all" that aren't in "train"
test_indices = np.setdiff1d(all_indices, train_indices)

X_train = X[train_indices]
Y_train = Y[train_indices]
X_test = X[test_indices]
Y_test = Y[test_indices]


# Now we're ready to set up our model

# Models using the Sequential API start like this
model = Sequential()

# You must specify the input shape into your first layer.
#  After that, keras can infer it for you
model.add(Dense(3, activation='relu', input_dim=4))


# What you do in the intermediate layers is up to you...
# The final layer needs to map closely to your output
# For binary classification, if should be a single sigmoid output.
# For multi-class classificaiton, it should match num_classes (and be sigmoid).
# For regression, it should be size one.
model.add(Dense(1, activation='sigmoid'))

# Let's look at the model we are going to estimate
model.summary()

# Every neuron has:
#  1 bias
#  number of weights = to the number of in-bound connections


# We now need to specify some options for how the stochastic gradient descent will proceed
# More details on this at:
# TODO: reference above
optimizer_details = SGD(lr=0.01, decay=1e-6)
# lr is the *learning rate* or how much to update the model at each step
# decay is the extent to dampen the learning rate each time
# idea is that as you get closer to the minimum of the function, you move more slowly


# There are other, fancier optimizers that do things like:
#  Intelligently estimate how much to update each parameter

# Now, we can compile our model by specifying a loss function, and passing our optimizer information
model.compile(loss='binary_crossentropy',
              optimizer=optimizer_details,
              metrics=['accuracy'])

# To fit our model, we simply pass in the X and Y data, and tell it how many epochs to run
# Quick aside first
#  Mini-batch: a set of examples to pass to the update algo
#  Epoch: every sample in _train_ seen by the update algo once

model.fit(X_train, Y_train, epochs=20)

# TODO: figure out where to link output shape, loss function, and problem class (here or slides or both?)

# So, is our model GOOD?
# Usual classification metrics apply

predicted_probabilities = model.predict_proba(X_test)
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_train, predicted_probabilities)

# So, is our model good? In what sense?
# If our AUC was 0.98, would we be done?

# Nope. Let's talk about hyperparameters.


# re: overfitting, underfitting, regularization
# http://playground.tensorflow.org/
