#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2.4
Ridge Regression
Machine Learning for Engineering Problem Solving
@author: Austin Downey
"""

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline

plt.close('all')

#%% build the data sets
m = 20
X = 6 * np.random.rand(m, 1) - 3
Y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

X_model = np.linspace(-3,3,num=1000)
X_model = np.expand_dims(X_model,axis=1)


#%% Perform Ridge Regression 

# plot the data
plt.figure()
plt.grid(True)
plt.scatter(X,Y,color='gray')
plt.xlabel('x')
plt.ylabel('y')

# build and plot a linear model 
model_linear = sk.linear_model.Ridge(alpha=100, solver="cholesky")
model_linear.fit(X, Y)
y_model_linear = model_linear.predict(X_model)
plt.plot(X_model,y_model_linear,'-',label='linear model')

# build and plot a polynomial model 
model_poly = sk.pipeline.make_pipeline(sk.preprocessing.PolynomialFeatures(10), 
                              sk.linear_model.Ridge(alpha=100, solver="cholesky"))
model_poly.fit(X, Y)
y_model_poly = model_poly.predict(X_model)
plt.plot(X_model,y_model_poly,'-',label='polynomial model')

plt.legend()





