#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3.2 Stochastic Gradient Descent (SDG) for the MINST data set
Created for EMCH 504 at USC
@author: Austin Downey
"""

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import datasets

cc = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
plt.close('all')

#%% Load your data

# Fetch the MNIST dataset from openml
mnist = sk.datasets.fetch_openml('mnist_784',as_frame=False,parser='auto') 
X = mnist['data']    # load the data
Y = np.asarray(mnist['target'],dtype=int)  # load the target

# Split the data set up into a training and testing data set
X_train = X[0:60000,:]
X_test = X[60000:,:]
Y_train = Y[0:60000]
Y_test = Y[60000:]

#%% Train a Stochastic Gradient Descent classifier

# Extract a subset for our "5-detector".
Y_train_5 = (Y_train == 5) 
Y_test_5 = (Y_test == 5)

# build and train the classifier
sgd_clf = sk.linear_model.SGDClassifier()
sgd_clf.fit(X_train, Y_train_5)

# get a digit from the dataset to test the classifier on
digit_id = 35 # An OK 5
# digit_id = 0 # An odd 5
# digit_id = 100 # A bad 5
test_digit = X[digit_id,:]
digit_reshaped = np.reshape(test_digit,(28,28))

# plot an image of the random pixel you picked above.
plt.figure()
plt.imshow(digit_reshaped,cmap = mpl.cm.binary,interpolation="nearest")
plt .title('A "'+str(Y[digit_id])+'" digit from the MNIST dataset')
plt.xlabel('pixel column number')
plt.ylabel('pixel row number')
plt.savefig('MNIST_digit')


# we can now test this for the "5" that we plotted earlier.
print(sgd_clf.predict([test_digit])) # a True case
print(sgd_clf.predict([X[2,:]])) # a False case 


