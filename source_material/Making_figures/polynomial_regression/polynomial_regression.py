#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open source notes/
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

#%% Load special modules 


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as PD
import scipy as sp
from scipy import interpolate
import pickle
import time
import re
import json as json
import pylab
import sklearn as sk
from sklearn import preprocessing
from sklearn import linear_model


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
plt.savefig('polynomial_regression_1',dpi=500)

#%% perform polynominal regression

# generate x^2 as we use the model y = x^2*a + x*b + b 
X_poly = np.vstack((X,X**2)).T


# or use the code as this does lots of features for multi-feature data sets. 
poly_features = sk.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
#X_poly = poly_features.fit_transform(np.expand_dims(X,axis=1))
X_poly = poly_features.fit_transform(X)

# In essence, we now have two data sets. We can plot that here
# plt.figure(figsize=(5,3))
# plt.grid(True)
# plt.scatter(X_poly[:,0],Y,label = 'data for x')
# plt.scatter(X_poly[:,1],Y,marker='s',label = 'data for $x^2$')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
# plt.savefig('polynomial_2',dpi=500)

# and fit linear models to these data sets
model = sk.linear_model.LinearRegression() # Select a linear model
model.fit(X_poly,Y) # Train the model
X_model = np.linspace(-3,3)


# the model paramaters are:
model_coefficients = model.coef_
model_intercept = model.intercept_
print(model_coefficients)
print(model_intercept)

# now if we plot the linear models on the extended set of features. 
plt.figure(figsize=(5,3))
plt.grid(True)
plt.scatter(X_poly[:,0],Y,label = 'data for $X$')
plt.scatter(X_poly[:,1],Y,marker='s',label = 'data for $X^2$')
plt.plot(X_model,model_coefficients[0][0]*X_model + model_intercept,'--',label='polynominal fit  $X$')
X_model_2 = np.linspace(0,9)
plt.plot(X_model_2,model_coefficients[0][1]*X_model_2 + model_intercept,':',label='linear fit for $X^2$',)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('polynomial_regression_2',dpi=500)


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
plt.savefig('polynomial_regression_3',dpi=500)


