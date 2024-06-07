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
# conf_mx = sk.metrics.confusion_matrix(Y_train, Y_train_pred)


#%% Extract 3s classifed as 3s, etc. 

threes_classified_as_three = []
i=0
while len(threes_classified_as_three) < 25:
    if Y_train[i]==3 and Y_train_pred[i]==3:
        threes_classified_as_three.append(i)
    i=i+1

threes_classified_as_five = []
i=0
while len(threes_classified_as_five) < 25:
    if Y_train[i]==3 and Y_train_pred[i]==5:
        threes_classified_as_five.append(i)
    i=i+1

five_classified_as_three = []
i=0
while len(five_classified_as_three) < 25:
    if Y_train[i]==5 and Y_train_pred[i]==3:
        five_classified_as_three.append(i)
    i=i+1

five_classified_as_five = []
i=0
while len(five_classified_as_five) < 25:
    if Y_train[i]==5 and Y_train_pred[i]==5:
        five_classified_as_five.append(i)
    i=i+1


#%% a check plot 

digit_id = 132
test_digit = X[digit_id,:]
digit_reshaped = np.reshape(test_digit,(28,28))

# plot an image of the random pixel you picked above.
plt.figure()
plt.imshow(digit_reshaped,cmap = mpl.cm.binary,interpolation="nearest")
plt.xlabel('pixel column number')
plt.ylabel('pixel row number')




#%% plot 5 x 5 figures 

selected_images = []
for digit in range(25):
    selected_images.append(X[threes_classified_as_three[digit]])
    
fig, axes = plt.subplots(5, 5, figsize=(3, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(selected_images[i],(28,28)), cmap = mpl.cm.binary)
    ax.axis('off')
plt.savefig('threes_classified_as_three.jpg',dpi=250)


selected_images = []
for digit in range(25):
    selected_images.append(X[threes_classified_as_five[digit]])
    
fig, axes = plt.subplots(5, 5, figsize=(3, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(selected_images[i],(28,28)), cmap = mpl.cm.binary)
    ax.axis('off')
plt.savefig('threes_classified_as_five.jpg',dpi=250)


selected_images = []
for digit in range(25):
    selected_images.append(X[five_classified_as_three[digit]])
    
fig, axes = plt.subplots(5, 5, figsize=(3, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(selected_images[i],(28,28)), cmap = mpl.cm.binary)
    ax.axis('off')
plt.savefig('five_classified_as_three.jpg',dpi=250)


selected_images = []
for digit in range(25):
    selected_images.append(X[five_classified_as_five[digit]])
    
fig, axes = plt.subplots(5, 5, figsize=(3, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(selected_images[i],(28,28)), cmap = mpl.cm.binary)
    ax.axis('off')
plt.savefig('five_classified_as_five.jpg',dpi=250)












