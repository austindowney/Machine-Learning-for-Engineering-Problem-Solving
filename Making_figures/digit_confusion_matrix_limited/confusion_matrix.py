#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3.7 Multiclass confusion matrix for the MINST data set

Developed for Machine Learning for Mechanical Engineers at the University of
South Carolina

@author: austin_downey
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

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

# set default fonts and plot colors
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.cursive':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.close('all')
#%% Load your data

# Fetch the MNIST dataset from openml
mnist = sk.datasets.fetch_openml('mnist_784',as_frame=False) 
X = np.asarray(mnist['data'])    # load the data and convert to np array
Y = np.asarray(mnist['target'],dtype=int)  # load the target

# Split the data set up into a training and testing data set
N = 1000
N_train = int(N*0.8)
N_val = N-N_train
X_train = X[0:N_train,:]
X_test = X[N_val:N,:]
Y_train = Y[0:N_train]
Y_test = Y[N_val:N]

digit_id = 1
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


######################## This code does it without k-fold 
# ovo_clf.fit(X_scaled,Y_train)
# Y_train_pred = ovo_clf.predict(X_scaled)
###############################################################################


conf_mx = sk.metrics.confusion_matrix(Y_train, Y_train_pred)
print(conf_mx)

# plot the results
fig = plt.figure(figsize=(4,4))
pos = plt.imshow(conf_mx,cmap=plt.cm.Greens,vmax=15) #, cmap=plt.cm.gray)
for i in range(10):
    for ii in range(10):
        if i==ii:
            plt.text(ii,i+0.2,conf_mx[i,ii],color='white',weight='bold',horizontalalignment='center')
        else:
            plt.text(ii,i+0.2,conf_mx[i,ii],color='black',weight='bold',horizontalalignment='center')

plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.yticks([0,1,2,3,4,5,6,7,8,9])
plt.ylabel('actual digit')
plt.xlabel('estimated digit')

plt.savefig('digit_confusion_matrix_limited',dpi=300)




























