#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:05:28 2019

Life satisfaction index example

@author: austin
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
from sklearn import linear_model
from sklearn import pipeline

# set default fonts and plot colors
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.close('all')

#%% build the data sets
np.random.seed(2) # 2 and 6 are pretty good
m = 100
X = 6 * np.random.rand(m,1) - 3
Y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)

#%% perform polynominal regression

# generate x^2 as we use the model y = x^2*a + x*b + b 
X_poly = np.vstack((X,X**2)).T

# or use the code as this does lots of features for multi-feature data sets. 
poly_features = sk.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
#X_poly = poly_features.fit_transform(np.expand_dims(X,axis=1))
X_poly = poly_features.fit_transform(X)

# and fit linear models to these data sets
model = sk.linear_model.LinearRegression() # Select a linear model
model.fit(X_poly,Y) # Train the model
X_model = np.linspace(-3,3)


#%% generate learing curves for a linear model

model = sk.linear_model.LinearRegression()

# split the data into training and validation data sets
# Split arrays or matrices into random train and test subsets
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, Y, test_size=0.2)

x_model = np.linspace(-3,3)

train_errors, val_errors = [], []
for i in range(1, len(X_train)):
    model.fit(X_train[:i], y_train[:i])
    y_train_predict = model.predict(X_train[:i])
    y_val_predict = model.predict(X_val)
    train_errors.append(sk.metrics.mean_squared_error(y_train_predict, y_train[:i]))
    val_errors.append(sk.metrics.mean_squared_error(y_val_predict, y_val))
    
plt.figure(figsize=(5,3))
plt.grid(True)
plt.plot(np.sqrt(train_errors), "--",label="training data")
plt.plot(np.sqrt(val_errors), ":", label="validation data")
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.legend(framealpha=1)
plt.ylim(0,3)
plt.tight_layout()
plt.savefig('overfitting_2',dpi=500)
#%% generate learning curves for a polynomial model degree=20

model = sk.pipeline.Pipeline((
("poly_features", sk.preprocessing.PolynomialFeatures(degree=20, include_bias=False)),
("lin_reg", sk.linear_model.LinearRegression()),
))


# split the data into training and validation data sets
# Split arrays or matrices into random train and test subsets
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, Y, test_size=0.2)

x_model = np.linspace(-3,3)

train_errors, val_errors = [], []
for i in range(1, len(X_train)):
    model.fit(X_train[:i], y_train[:i])
    y_train_predict = model.predict(X_train[:i])
    y_val_predict = model.predict(X_val)
    train_errors.append(sk.metrics.mean_squared_error(y_train_predict, y_train[:i]))
    val_errors.append(sk.metrics.mean_squared_error(y_val_predict, y_val))
    
plt.figure(figsize=(5,3))
plt.grid(True)
plt.plot(np.sqrt(train_errors), "--",label="training data")
plt.plot(np.sqrt(val_errors), ":", label="validation data")
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.legend(framealpha=1)
plt.ylim(0,3)
plt.tight_layout()
plt.savefig('overfitting_3',dpi=500)


#%% generate learning curves for a polynomial model degree=2

model = sk.pipeline.Pipeline((
("poly_features", sk.preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
("lin_reg", sk.linear_model.LinearRegression()),
))


# split the data into training and validation data sets
# Split arrays or matrices into random train and test subsets
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, Y, test_size=0.2)

x_model = np.linspace(-3,3)

train_errors, val_errors = [], []
for i in range(1, len(X_train)):
    model.fit(X_train[:i], y_train[:i])
    y_train_predict = model.predict(X_train[:i])
    y_val_predict = model.predict(X_val)
    train_errors.append(sk.metrics.mean_squared_error(y_train_predict, y_train[:i]))
    val_errors.append(sk.metrics.mean_squared_error(y_val_predict, y_val))
    
plt.figure(figsize=(5,3))
plt.grid(True)
plt.plot(np.sqrt(train_errors), "--",label="training data")
plt.plot(np.sqrt(val_errors), ":", label="validation data")
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.legend(framealpha=1)
plt.ylim(0,3)
plt.tight_layout()
plt.savefig('overfitting_4',dpi=500)
