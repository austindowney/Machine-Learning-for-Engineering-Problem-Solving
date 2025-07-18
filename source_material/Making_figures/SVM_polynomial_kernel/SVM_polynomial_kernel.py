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
import matplotlib.pyplot as plt


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

import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn import svm


#%% SVM polynominal features




X, y = make_moons(n_samples=100, noise=0.15, random_state=2)




poly_kernel_svm_clf = Pipeline([
        ("scaler", sk.preprocessing.StandardScaler()),
        ("svm_clf", sk.svm.SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)


poly100_kernel_svm_clf = Pipeline([
        ("scaler", sk.preprocessing.StandardScaler()),
        ("svm_clf", sk.svm.SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)


#%% Plot the figure

axes = [-1.5, 2.4, -1, 1.5]


plt.figure(figsize=(6.5,2.5))

ax1 = plt.subplot(121)
x0s = np.linspace(axes[0], axes[1], 100)
x1s = np.linspace(axes[2], axes[3], 100)
x0, x1 = np.meshgrid(x0s, x1s)
X2 = np.c_[x0.ravel(), x1.ravel()]
y_pred = poly_kernel_svm_clf.predict(X2).reshape(x0.shape)
y_decision = poly_kernel_svm_clf.decision_function(X2).reshape(x0.shape)
plt.contourf(x0, x1, y_pred, cmap="Pastel2")
#plt.contourf(x0, x1, y_decision, cmap="Pastel2", alpha=0.1)
    
#plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", markersize=3)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "d", markersize=3)
plt.axis(axes)
plt.grid(True)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title(r"$degree=3$ and $coef0=1$")


ax2 = plt.subplot(122)
x0s = np.linspace(axes[0], axes[1], 100)
x1s = np.linspace(axes[2], axes[3], 100)
x0, x1 = np.meshgrid(x0s, x1s)
X2 = np.c_[x0.ravel(), x1.ravel()]
y_pred = poly100_kernel_svm_clf.predict(X2).reshape(x0.shape)
y_decision = poly100_kernel_svm_clf.decision_function(X2).reshape(x0.shape)
plt.contourf(x0, x1, y_pred, cmap="Pastel2")
#plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", markersize=3)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "d", markersize=3)
plt.axis(axes)
plt.grid(True)
plt.xlabel("$x_1$")

ax2.set_yticklabels([])
plt.title(r"$degree=10$ and $coef0=100$")

plt.tight_layout()
plt.savefig("SVM_polynomial_kernel",dpi=300)



