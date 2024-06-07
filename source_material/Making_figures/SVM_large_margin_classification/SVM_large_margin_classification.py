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

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors=cc[6])
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

#%% Large margin classification



iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classifier model
svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X, y)   
    
# Bad models
x0 = np.linspace(0, 5.5, 200)
pred_1 = x0 - 1.8
pred_2 = 0.1 * x0 + 0.5



plt.figure(figsize=(6.5,2.75))

plt.subplot(121)
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", label="Setosa")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "s", label="Versicolor")
plt.plot(x0, pred_1, "--",)
plt.plot(x0, pred_2, "--",)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend(loc="upper left")
plt.axis([0, 5.5, 0, 2])

ax2 = plt.subplot(122)
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "s")
plt.xlabel("petal length (cm)")
plt.axis([0, 5.5, 0, 2])

ax2.set_yticklabels([])

plt.tight_layout()
plt.savefig("SVM_large_margin_classification",dpi=300)




