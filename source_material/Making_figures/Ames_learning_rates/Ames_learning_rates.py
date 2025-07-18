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
plt.rcParams.update({'font.size': 9})
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
import pandas as pd
import seaborn as sns

plt.close('all')




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

price_model_X = np.linspace(0,6000)

# add a dimension to the data as math is easier in 2d arrays and sk learn only 
# takes 2d arrays
X = np.expand_dims(X,axis=1)
Y = np.expand_dims(Y,axis=1)
price_model_X = np.expand_dims(price_model_X,axis=1)

# compute 
X_b = np.ones((X.shape[0],2))
X_b[:,1] = X.T # add x0 = 1 to each instance


#%% compute the linear regression solution using gradient descent

plt.figure(figsize=(6.5,3.))
plt.subplot(1,3,1)
eta = 0.0000005 # learning rate
n_iterations = 10
m = X.shape[0]
beta_gradient_descent = np.array([[0],[0]]) #np.random.randn(2,1) # random initialization
gradient_descents = []
gradient_descents.append(np.copy(beta_gradient_descent))
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(beta_gradient_descent) - Y)
    beta_gradient_descent = beta_gradient_descent - eta * gradients
    gradient_descents.append(np.copy(beta_gradient_descent))
price_model_Y_gradient_descent =  beta_gradient_descent[0]+beta_gradient_descent[1]*price_model_X

plt.plot(X,Y,'o',markersize=1,label='data points')
plt.xlabel('above grade (ground)\nliving area square feet')
plt.ylabel('price (USD thousands)')
plt.ylim([-800,800])
plt.xlim([0,5000])
#plt.plot(price_model_X,price_model_Y_gradient_descent,':',label='gradient descent')
print(gradient_descents)
for i in range(n_iterations):
    price_model_Y_gradient_descent =  gradient_descents[i][0]+gradient_descents[i][1]*price_model_X
    plt.plot(price_model_X,price_model_Y_gradient_descent,'-')
plt.grid(True)
plt.title('$\eta = $'+str(eta))




plt.subplot(1,3,2)
eta = 0.00000005 # learning rate
n_iterations = 10
m = X.shape[0]
beta_gradient_descent = np.array([[0],[0]]) #np.random.randn(2,1) # random initialization
gradient_descents = []
gradient_descents.append(np.copy(beta_gradient_descent))
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(beta_gradient_descent) - Y)
    beta_gradient_descent = beta_gradient_descent - eta * gradients
    gradient_descents.append(np.copy(beta_gradient_descent))
price_model_Y_gradient_descent =  beta_gradient_descent[0]+beta_gradient_descent[1]*price_model_X

plt.plot(X,Y,'o',markersize=1,label='data points')
plt.xlabel('above grade (ground)\nliving area square feet')
plt.ylim([-0,800])
plt.xlim([0,5000])
#plt.plot(price_model_X,price_model_Y_gradient_descent/1000,':',label='gradient descent')
for i in range(n_iterations):
    price_model_Y_gradient_descent =  gradient_descents[i][0]+gradient_descents[i][1]*price_model_X
    plt.plot(price_model_X,price_model_Y_gradient_descent,'-')
plt.grid(True)
plt.title('$\eta = $'+str(eta))


plt.subplot(1,3,3)
eta = 0.000000005 # learning rate
n_iterations = 10
m = X.shape[0]
beta_gradient_descent = np.array([[0],[0]]) #np.random.randn(2,1) # random initialization
gradient_descents = []
gradient_descents.append(np.copy(beta_gradient_descent))
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(beta_gradient_descent) - Y)
    beta_gradient_descent = beta_gradient_descent - eta * gradients
    gradient_descents.append(np.copy(beta_gradient_descent))
price_model_Y_gradient_descent =  beta_gradient_descent[0]+beta_gradient_descent[1]*price_model_X

plt.plot(X,Y,'o',markersize=1,label='data points')
plt.xlabel('above grade (ground)\nliving area square feet')
plt.ylim([0,800])
plt.xlim([0,5000])
#plt.plot(price_model_X,price_model_Y_gradient_descent/1000,':',label='gradient descent')
for i in range(n_iterations):
    price_model_Y_gradient_descent =  gradient_descents[i][0]+gradient_descents[i][1]*price_model_X
    plt.plot(price_model_X,price_model_Y_gradient_descent,'-')
plt.grid(True)
plt.title('$\eta = $'+str(eta))








plt.tight_layout()
plt.savefig('Ames_learning_rates',dpi=500)

