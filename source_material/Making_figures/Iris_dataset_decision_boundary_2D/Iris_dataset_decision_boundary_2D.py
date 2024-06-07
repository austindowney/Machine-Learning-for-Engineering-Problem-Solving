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
import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import datasets

#%% Load your data

# We will use the Iris data set.  This dataset was created by biologist Ronald
# Fisher in his 1936 paper "The use of multiple measurements in taxonomic 
# problems" as an example of linear discriminant analysis

iris = sk.datasets.load_iris()

# for simplicity, extract some of the dats sets
data = iris['data'] # this contains the length of the pedals and sepals
target = iris['target'] # containt what type of flower it is
target_names = iris['target_names'] # containt the name that aligns with the type of the flower
feature_names = iris['feature_names'] # the names of the features


#%% Train a Logistic Regression model

# define the features (X) and the label (Y)
X = iris["data"][:, 3:] # consider just the petal width
y = (iris["target"] == 2).astype(int) # 1 if Iris-Virginica, else 0

# Build the logistic Regression model and train it. 
log_reg = sk.linear_model.LogisticRegression()
log_reg.fit(X, y)

# Build a range of the feature (X) to predict over. Here we just consider pedal width/
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)

# Use the Logistic Refression Model to predict the pedal type based on pedal width
y_proba = log_reg.predict_proba(X_new)


#%% plot the Linear decision boundary in 2D "Pedal" space that considers 

# build the training and target set.
X = data[:, (2, 3)]  # petal length, petal width
y = target == 2


# build the Logistic Regression model
log_reg = sk.linear_model.LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
# Note: The hyperparameter controlling the regularization strength of a Scikit-Learn 
# LogisticRegression model is not alpha (as in other linear models), but its 
# inverse: C. The higher the value of C, the less the model is regularized.


# train the Logistic Regression model
log_reg.fit(X, y)

# build the x values for the predections over the entire "pedal space"
x0, x1 = np.meshgrid(
        np.linspace(2.8, 7, 500),
        np.linspace(0.3, 3, 200),
    )
X_new = np.vstack((x0.reshape(-1), x1.reshape(-1))).T # build a vector format of the mesh grid

# predict on the vectorized format
y_predict = log_reg.predict(X_new)
y_proba = log_reg.predict_proba(X_new)

# convert back to meshgrid shape for plotting
zz_predict = y_predict.reshape(x0.shape)
zz_proba = y_proba[:, 1].reshape(x0.shape)

# plot the 2D "pedal space"
plt.figure(figsize=(6.5,3))
plt.grid(True)
plt.scatter(data[target==1,2],data[target==1,3],marker='s',color=cc[1],label=target_names[1],zorder=10)
plt.scatter(data[target==2,2],data[target==2,3],marker='d',color=cc[2],label=target_names[2],zorder=10)
plt.contourf(x0, x1, zz_predict, cmap='Pastel2')
contour = plt.contour(x0, x1, zz_proba, [0.100,0.5,0.900],cmap=plt.cm.brg,zorder=10)
plt.clabel(contour, inline=1, fontsize=12) # add the labels to the plot
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
l = plt.legend(framealpha=1)
l.set_zorder(10)
plt.tight_layout()
plt.savefig("Iris_dataset_decision_boundary_2D",dpi=400)






















