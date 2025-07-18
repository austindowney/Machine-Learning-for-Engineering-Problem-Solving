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

import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline


#%% build the data sets
m = 20
X = 6 * np.random.rand(m, 1) - 3
Y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

X_model = np.linspace(-3,3,num=1000)
X_model = np.expand_dims(X_model,axis=1)

#%% Perform linear regression using the Ridge Regression method using the closed form method

plt.figure(figsize=(6.5,3))
plt.subplot(121)
plt.grid(True)
plt.scatter(X,Y,label='data',color='gray')
line_style = ['-','--',':']
n=0
for i in [1,100,10000]:   # set the alpha paramaters. 
    model = sk.linear_model.Ridge(alpha=i) #sk.linear_model.LinearRegression()
    
    #sk.pipeline.make_pipeline(sk.preprocessing.PolynomialFeatures(10), sk.linear_model.Ridge(alpha=i, solver="cholesky"))
    model.fit(X, Y)
    y_model = model.predict(X_model)
    plt.plot(X_model,y_model,line_style[n],label=r'$\alpha=$'+str(i)); n=n+1
plt.xlabel('x\n(a)')
plt.ylabel('y')
plt.ylim([-1,11])
plt.legend(framealpha=1)
plt.title('linear ridge regression')


#%% Perform polynomial regression using the Ridge Regression method using the closed form method

plt.subplot(122)
plt.grid(True)
plt.scatter(X,Y,label='data',color='gray')
line_style = ['-','--',':']
n=0
for i in [0,100,10000]:   # set the alpha paramaters. 
    model = sk.pipeline.make_pipeline(sk.preprocessing.PolynomialFeatures(10), 
                                  sk.linear_model.Ridge(alpha=i, solver="cholesky"))
    model.fit(X, Y)
    y_model = model.predict(X_model)
    plt.plot(X_model,y_model,line_style[n],label=r'$\alpha=$'+str(i)); n=n+1
plt.xlabel('x\n(b)')
plt.ylabel('y')
plt.ylim([-1,11])
plt.legend(framealpha=1)
plt.title('polynomial ridge regression')

plt.tight_layout(pad=0)
plt.savefig('ridge_regression',dpi=300)


