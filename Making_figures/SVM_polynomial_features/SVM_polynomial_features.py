#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

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


import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
import time as time
from sklearn import linear_model
from sklearn import pipeline
from sklearn import datasets
from sklearn import multiclass
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#%% SVM polynominal features



    

X, y = make_moons(n_samples=100, noise=0.15, random_state=2)



polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", sk.preprocessing.StandardScaler()),
        ("svm_clf", sk.svm.LinearSVC(C=10, loss="hinge", random_state=2))
    ])

polynomial_svm_clf.fit(X, y)

plt.figure(figsize=(6.5,3))

axes = [-1.5, 2.5, -1, 1.5]
x0s = np.linspace(axes[0], axes[1], 100)
x1s = np.linspace(axes[2], axes[3], 100)
x0, x1 = np.meshgrid(x0s, x1s)
X2 = np.c_[x0.ravel(), x1.ravel()]
y_pred = polynomial_svm_clf.predict(X2).reshape(x0.shape)
y_decision = polynomial_svm_clf.decision_function(X2).reshape(x0.shape)
plt.contourf(x0, x1, y_pred, cmap="Pastel2", alpha=1)
#plt.contour(x0, x1, y_decision, cmap="Pastel2", alpha=0.1)
#contour = plt.contour(x0, x1, y_pred, [0.100,0.5,0.900], cmap=plt.cm.brg)

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "d")
plt.axis(axes)
plt.grid(True)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.tight_layout()
plt.savefig("SVM_polynomial_features")




























