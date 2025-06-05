#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2.5
Early Stopping
Machine Learning for Engineering Problem Solving
@author: Austin Downey
"""

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk


plt.close('all')

#%% build the data sets

# use 6 to help give a smooth curve that makes the case for early stopping
np.random.seed(6) 

m = 20
X = 6 * np.random.rand(m, 1) - 3
Y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# plot the data
plt.figure()
plt.grid(True)
plt.scatter(X,Y,color='gray')
plt.xlabel('x')
plt.ylabel('y')

X_model = np.linspace(-3,3,num=1000)
X_model = np.expand_dims(X_model,axis=1)

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, Y, test_size=0.2)

#%% perform early stopping


# prepare the data
poly_scaler = sk.pipeline.Pipeline([("poly_features", sk.preprocessing.PolynomialFeatures(
    degree=90, include_bias=False)), ("std_scaler", sk.preprocessing.StandardScaler())])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.fit_transform(X_val)

# set up the model, not that by setting max_iter=1 it will only train one epoch
model = sk.linear_model.SGDRegressor(max_iter=1, tol=0,learning_rate="constant"
    ,eta0=0.0005,penalty=None,warm_start=True)


# Train the model in a loop to build the data set to investigate the benefit of early stopping
val_errors = []
train_errors = []
for epoch in range(1000):
    model.fit(X_train_poly_scaled, y_train.ravel()) # continues where it left off
    y_val_predict = model.predict(X_val_poly_scaled) # Predict the target values
    y_train_predict = model.predict(X_train_poly_scaled) # Predict the target values
    val_error = sk.metrics.mean_squared_error(y_val, y_val_predict)  # Calculate error
    train_error = sk.metrics.mean_squared_error(y_train.ravel(), y_train_predict)  # Calculate error
    val_errors.append(val_error)
    train_errors.append(train_error)

# plot the early learning curves, you may have to plot this a few times to get 
# a set of curves that shows strong results
plt.figure()
plt.grid(True)
plt.plot(val_errors,label='validation data')
plt.plot(train_errors,'--',label='training data')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend()





