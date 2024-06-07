#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.

Header file last updated May 16, 2024
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl


# set default fonts and plot colors
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
# I don't think I need this next line as its set to 'stixsans' above. 
plt.rcParams.update({'mathtext.fontset': 'custom'}) 
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Load special modules as needed


import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
import time as time
from sklearn import linear_model
from sklearn import pipeline
from sklearn import datasets
from sklearn import multiclass


#%% Load your data

# Fetch the MNIST dataset from openml
mnist = sk.datasets.fetch_openml('mnist_784',as_frame=False,parser='auto') 
X = np.asarray(mnist['data'])    # load the data and convert to np array
Y = np.asarray(mnist['target'],dtype=int)  # load the target

# Split the data set up into a training and testing data set
X_train = X[0:60000,:]
X_test = X[60000:,:]
Y_train = Y[0:60000]
Y_test = Y[60000:]

digit_id = 5151
test_digit = X[digit_id,:]
digit_reshaped = np.reshape(test_digit,(28,28))

# plot an image of the random pixel you picked above.
plt.figure()
plt.imshow(digit_reshaped,cmap = mpl.cm.binary,interpolation="nearest")
plt.title('A "'+str(Y[digit_id])+'" digit from the MNIST dataset')
plt.xlabel('pixel column number')
plt.ylabel('pixel row number')

#%% Confusion Matrix for a Multiclass classifier. 

# Use the one-vs-one classifier that uses Stochastic Gradient Descent as this is 
# faster for this specific data set

# here we test a 
ovo_clf = sk.multiclass.OneVsOneClassifier(sk.linear_model.SGDClassifier())

# scale the data using sklean standard scaler. This standardize features by
# removing the mean and scaling to unit variance
scaler = sk.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X_train.astype(np.float64))

# make a prediction for every case using the k-fold method. 
Y_train_pred = sk.model_selection.cross_val_predict(ovo_clf, X_scaled, Y_train, cv=3)
conf_mx = sk.metrics.confusion_matrix(Y_train, Y_train_pred)
print(conf_mx)

# plot the results
fig = plt.figure(figsize=(4,4))
pos = plt.imshow(conf_mx) #, cmap=plt.cm.gray)
cbar = plt.colorbar(pos)
cbar.set_label('number of classified digits')
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.yticks([0,1,2,3,4,5,6,7,8,9])
plt.ylabel('actual digit')
plt.xlabel('estimated digit')
plt.savefig('digit_confusion_matrix',dpi=300)

# normalize the data as to compare error rates so classes with more data don't look bad.
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums


# remove the high values along the diagonal by replacing them with NANs
conf_mx_noise = np.asarray(norm_conf_mx,dtype=np.float32)
np.fill_diagonal(conf_mx_noise, np.NaN)


# plot the results but only consider the noise
fig = plt.figure(figsize=(4,4))
pos = plt.imshow(conf_mx_noise) #, cmap=plt.cm.gray)
cbar = plt.colorbar(pos)
cbar.set_label('normalized classification error')
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.yticks([0,1,2,3,4,5,6,7,8,9])
plt.ylabel('actual digit')
plt.xlabel('estimated digit')
plt.savefig('digit_confusion_matrix_error',dpi=300)


