#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().magic('reset -sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open source notes/
books. This common header is used to set the fonts and format.

Header file last updated March 10, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

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
plt.rcParams.update({'mathtext.fontset': 'custom'}) # I don't think I need this as its set to 'stixsans' above.
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Load the modules needed for this code. 


import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import datasets


#%% load data



ames = sk.datasets.fetch_openml(name="house_prices", as_frame=True)
target = ames['target'].values
data = ames['data']
YrSold = data['YrSold'].values  # Year Sold (YYYY)
MoSold = data['MoSold'].values  # Month Sold (MM)
OverallCond = data['OverallCond'].values # OverallCond: Rates the overall condition of the house
GrLivArea = data['GrLivArea'].values # Above grade (ground) living area square feet


# Ask a home buyer to describe their dream house, and they probably won't begin 
# with the height of the basement ceiling or the proximity to an east-west railroad. 
# But this playground competition's dataset proves that much more influences price 
# negotiations than the number of bedrooms or a white-picket fence.

# With 79 explanatory variables describing (almost) every aspect of residential 
# homes in Ames, Iowa, this competition challenges you to predict the final price 
# of each home.


#%% Build a model for the data

X = GrLivArea
Y = target/1000
price_model_X = np.linspace(0,5000)

theta_0 = 0
theta_1 = 0.130
price_model_Y_manual = theta_0 + theta_1*price_model_X

# plot the modeled results
plt.figure(figsize=(5,3))
plt.plot(X,Y,'o',markersize=2,label='data')
plt.xlabel('above grade (ground) living area square feet')
plt.ylabel('price (USD thousands)')
plt.grid(True)
plt.ylim([0,800])
plt.xlim([0,6000])
plt.tight_layout()
plt.savefig('Ames_simple_linear_regression_model_1',dpi=500)

# add a dimension to the data as sk-learn only takes 2-d arrays
X = np.expand_dims(X,axis=1)
Y = np.expand_dims(Y,axis=1)
price_model_X = np.expand_dims(price_model_X,axis=1)

# run the model in sk-learn
model = sk.linear_model.LinearRegression()
model.fit(X,Y)
price_model_Y_sklearn = model.predict(price_model_X)

# plot the modeled results
plt.figure(figsize=(5,3))
plt.plot(X,Y,'o',markersize=2,label='data')
plt.plot(price_model_X,price_model_Y_manual,'r--',label='manual fit')
plt.plot(price_model_X,price_model_Y_sklearn,'g:',label='sklearn fit')
plt.xlabel('above grade (ground) living area square feet')
plt.ylabel('price (USD thousands)')
plt.grid(True)
plt.legend(framealpha=1)
plt.ylim([0,800])
plt.xlim([0,6000])
plt.tight_layout()
plt.savefig('Ames_simple_linear_regression_model_2',dpi=500)


#%% compute the linear regression solution using the closed form solution

# compute 
X_b = np.ones((X.shape[0],2))
X_b[:,1] = X.T # add x0 = 1 to each instance

beta_closed_form = np.linalg.inv(X_b.T@X_b)@X_b.T@Y

price_model_y_closed_form = beta_closed_form[0] + beta_closed_form[1]*4000
price_model_Y_closed_form = beta_closed_form[0] + beta_closed_form[1]*price_model_X

plt.figure(figsize=(5,3))
plt.plot(X,Y,'o',markersize=3,label='data points')
plt.xlabel('above grade (ground) living area square feet')
plt.ylabel('price (USD thousands)')
plt.plot(4000,price_model_y_closed_form,'dr',markersize=10,zorder=10,label='inferred data point')
plt.plot(price_model_X,price_model_Y_closed_form,'--',label='closed form')
plt.grid(True)
plt.legend(framealpha=1)
plt.ylim([0,800])
plt.xlim([0,6000])
plt.tight_layout()
plt.savefig('Ames_simple_linear_regression_model_3',dpi=500)





























