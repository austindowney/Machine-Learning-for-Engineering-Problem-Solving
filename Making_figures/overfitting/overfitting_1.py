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


#%% perform polynominal regression


x_model = np.linspace(-3,3)

plt.figure(figsize=(5,3))
plt.plot(X,Y,'o',markersize=3,label='data')

model = sk.pipeline.Pipeline((
("poly_features", sk.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),
("lin_reg", sk.linear_model.LinearRegression()),
))
model.fit(X, Y)
plt.plot(x_model,model.predict(np.expand_dims(x_model,axis=1)),'--',label='degree=1')


model = sk.pipeline.Pipeline((
("poly_features", sk.preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
("lin_reg", sk.linear_model.LinearRegression()),
))
model.fit(X, Y)
plt.plot(x_model,model.predict(np.expand_dims(x_model,axis=1)),':',label='degree=2')

model = sk.pipeline.Pipeline((
("poly_features", sk.preprocessing.PolynomialFeatures(degree=30, include_bias=False)),
("lin_reg", sk.linear_model.LinearRegression()),
))
model.fit(X, Y)
plt.plot(x_model,model.predict(np.expand_dims(x_model,axis=1)),'-.',label='degree=30')



plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=9,framealpha=1,ncol=2)
plt.ylim(-2,12)
plt.tight_layout()
plt.grid(True)
plt.savefig('overfitting_1',dpi=500)
    
    

