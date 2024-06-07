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

#X = 6 * np.random.rand(m) - 3
#Y = 0.5 * X**2 + X + 2 + np.random.randn(m)

# plot the data
plt.figure(figsize=(5,3))
plt.grid(True)
plt.scatter(X,Y)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('polynomial_1',dpi=500)

#%% perform polynominal regression

X_model = np.linspace(-3,3)


plt.figure(figsize=(5,3))
plt.scatter(X,Y,s=2,label='data')

model = sk.pipeline.Pipeline((
("poly_features", sk.preprocessing.PolynomialFeatures(degree=30, include_bias=False)),
("lin_reg", sk.linear_model.LinearRegression()),
))



model.fit(X, Y)
y_train_predict = model.predict(np.expand_dims(X_model,axis=1))

x_model = np.linspace(-3,3)



#model.fit(X_train, y_train)
model.fit(X, Y)



plt.plot(x_model,model.predict(np.expand_dims(x_model,axis=1)),'r--',label='model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=2)
plt.ylim(-2,12)
plt.grid(True)

    
    
    
    
    
    


# plt.figure('test model')
# plt.scatter(X,Y,s=2,label='data')
# plt.scatter(X_train[:i],y_train[:i],label='data in training set')
# plt.scatter(X_val,y_val,marker='s',label='data in validation set')
# plt.plot(x_model,model.predict(np.expand_dims(x_model,axis=1)),'r--',label='model')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=2)
# plt.ylim(-2,12)
# plt.grid(True)
# plt.savefig('test_plots/polynominal_model'+str(i))
# plt.close('test model')
    
    
    
    



# generate x^2 as we use the model y = x^2*a + x*b + b 
X_poly = np.vstack((X,X**2)).T


# or use the code as this does lots of features for multi-feature data sets. 
poly_features = sk.preprocessing.PolynomialFeatures(degree=30, include_bias=False)
#X_poly = poly_features.fit_transform(np.expand_dims(X,axis=1))
X_poly = poly_features.fit_transform(X)


# and fit linear models to these data sets
model = sk.linear_model.LinearRegression() # Select a linear model
model.fit(X_poly,Y) # Train the model



# the model paramaters are:
model_coefficients = model.coef_
model_intercept = model.intercept_
print(model_coefficients)
print(model_intercept)

# now if we plot the linear models on the extended set of features. 
plt.figure(figsize=(5,3))
plt.grid(True)
plt.scatter(X_poly[:,0],Y,label = 'data for x')
plt.scatter(X_poly[:,1],Y,marker='s',label = 'data for $x^2$')
plt.plot(X_model,model_coefficients[0][0]*X_model + model_intercept,'--',label='polynominal fit  $x$')
X_model_2 = np.linspace(0,9)
plt.plot(X_model_2,model_coefficients[0][1]*X_model_2 + model_intercept,':',label='linear fit for $x^2$',)
plt.plot(X_model,model.predict(np.expand_dims(X_model,axis=1)))
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('polynomial_2',dpi=500)


# now that we have a parameter for x and x^2, these can be recombined into a single
# model, y = x^2*a + x*b + b.

plt.figure(figsize=(5,3))
plt.grid(True)
plt.scatter(X,Y,label='data')
plt.plot(X_model,model_coefficients[0][1]*X_model**2 + model_coefficients[0][0]*X_model + 
         model_intercept,'r--',label='polynominal fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.savefig('polynomial_3',dpi=500)




# plt.figure(figsize=(6.5,4))
# plt.subplot(121)
# plt.grid(True)
# plt.scatter(X_poly[:,0],Y,label = 'data for x')
# plt.scatter(X_poly[:,1],Y,marker='s',label = 'data for $x^2$')
# plt.plot(X_model,model_coefficients[0][0]*X_model + model_intercept,'--',label='polynominal fit  $x$')
# X_model_2 = np.linspace(0,9)
# plt.plot(X_model_2,model_coefficients[0][1]*X_model_2 + model_intercept,':',label='linear fit for $x^2$',)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.subplot(122)
# plt.grid(True)
# plt.scatter(X,Y,label='data')
# plt.plot(X_model,model_coefficients[0][1]*X_model**2 + model_coefficients[0][0]*X_model + 
#          model_intercept,'r--',label='polynominal fit')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.tight_layout()
# plt.savefig('polynomial_4',dpi=500)



#%% generate learing curves for a linear model

# #%% build the data sets
# m = 100
# X = 6 * np.random.rand(m) - 3
# Y = 0.5 * X**2 + X + 2 + np.random.randn(m)


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
    
    # plt.figure('test model')
    # plt.scatter(X,Y,s=2,label='data')
    # plt.scatter(X_train[:i],y_train[:i],label='data in training set')
    # plt.scatter(X_val,y_val,marker='s',label='data in validation set')
    # plt.plot(x_model,model.predict(np.expand_dims(x_model,axis=1)),'r--',label='model')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend(loc=2)
    # plt.ylim(-2,12)
    # plt.grid(True)
    # plt.savefig('test_plots/linear_model'+str(i))
    # plt.close('test model')

plt.figure(figsize=(5,3))
plt.grid(True)
plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.ylim(0,3)
plt.tight_layout()
plt.savefig('overfitting_2',dpi=500)
#%% generate learning curves for a polynomial model

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
    
    # plt.figure('test model')
    # plt.scatter(X,Y,s=2,label='data')
    # plt.scatter(X_train[:i],y_train[:i],label='data in training set')
    # plt.scatter(X_val,y_val,marker='s',label='data in validation set')
    # plt.plot(x_model,model.predict(np.expand_dims(x_model,axis=1)),'r--',label='model')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend(loc=2)
    # plt.ylim(-2,12)
    # plt.grid(True)
    # plt.savefig('test_plots/polynominal_model'+str(i))
    # plt.close('test model')

plt.figure(figsize=(5,3))
plt.grid(True)
plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
plt.xlabel('number of data points in training set')
plt.ylabel('mean squared error')
plt.ylim(0,3)
plt.tight_layout()
plt.savefig('overfitting_3',dpi=500)


