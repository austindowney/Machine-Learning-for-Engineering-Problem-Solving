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


#%% build the data sets
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

# set up the model, not tat by setting max_iter=1 it will only train one epoch
sgd_reg = sk.linear_model.SGDRegressor(max_iter=1, tol=0, warm_start=True,
                                       penalty=None, learning_rate="constant", 
                                       eta0=0.0005)


# Train the model in a loop to build the data set to investigate the benefit of early stopping
val_errors = []
train_errors = []
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train.ravel()) # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled) # Predict the target values
    y_train_predict = sgd_reg.predict(X_train_poly_scaled) # Predict the target values
    train_error = sk.metrics.mean_squared_error(y_train.ravel(), y_train_predict)  # Calculate error
    val_error = sk.metrics.mean_squared_error(y_val, y_val_predict)  # Calculate error
    val_errors.append(val_error)
    train_errors.append(train_error)

# plot the early learning curves, you may have to plot this a few times to get 
# a set of curves that shows strong results
plt.figure(figsize=(6,3))
plt.grid(True)
plt.plot(val_errors,label='validation data')
plt.plot(train_errors,'--',label='training data')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig('early_stopping_trail.pdf')





