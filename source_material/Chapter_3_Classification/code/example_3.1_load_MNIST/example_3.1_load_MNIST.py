#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3.1 Load the MNIST data set
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

plt.close('all')


#%% Load your data

# this fetches "a" MNIST dataset  from openml and loads it into your environment
# as a Bunch, a Dictionary-like object that exposes its keys as attributes.
mnist = sk.datasets.fetch_openml('mnist_784',as_frame=False,parser='auto') 

# calling the DESCR key will return a description of the dataset
print(mnist['DESCR'])

# calling the data key will return an array with one row per instance and one 
# column per feature where each features is a pixel, as defined in the key feature_names
X = mnist['data']


# calling the target key will return an array with the labels
Y = np.asarray(mnist['target'],dtype=int)

# Each image is 784 features or 28×28 pixels, however, the features must be reshaped
# into a 29x29 grid to make them into a digit, where the values represents one 
# the intensity of one pixel, from 0 (white) to 255 (black).

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

