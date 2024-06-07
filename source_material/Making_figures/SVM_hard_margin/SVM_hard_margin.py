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

import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
import time as time
from sklearn import linear_model
from sklearn import pipeline
from sklearn import datasets
from sklearn import multiclass
from sklearn.svm import SVC


#%% Large margin classification


np.random.seed(2)

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # Get the coefficients (weights) and intercept from the trained SVM model
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # Rearrange to solve for x1: x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)  # Generate 200 points from xmin to xmax
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]  # Calculate the decision boundary

    # Calculate the margin boundaries (distance from the decision boundary)
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    # Plot the decision boundary and margins
    plt.plot(x0, decision_boundary, "k-", linewidth=2)  # Decision boundary
    plt.plot(x0, gutter_up, "k--", linewidth=2)  # Margin boundary (upper)
    plt.plot(x0, gutter_down, "k--", linewidth=2)  # Margin boundary (lower)

#%% Sensitivity to outliers


# Define outlier data points
X_outliers = np.array([[4.2, 1.1], [3.27, 0.7]])
y_outliers = np.array([0, 0])

# Combine original data with the first outlier
Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
yo1 = np.concatenate([y, y_outliers[:1]], axis=0)

# Combine original data with the second outlier
Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
yo2 = np.concatenate([y, y_outliers[1:]], axis=0)

# Train SVM classifier with the second set of data including outliers
svm_clf2 = SVC(kernel="linear", C=10**9)
svm_clf2.fit(Xo2, yo2)




#%% Create a figure for the plots
plt.figure(figsize=(6.5, 3.0))

# Plot the first dataset with one outlier (left subplot)
ax1 = plt.subplot(121)
plt.plot(Xo1[:, 0][yo1 == 0], Xo1[:, 1][yo1 == 0], "o")  # Class 0 points
plt.plot(Xo1[:, 0][yo1 == 1], Xo1[:, 1][yo1 == 1], "s")  # Class 1 points
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.axis([0, 5.5, 0, 2])  # Set axis limits

# Plot the second dataset with another outlier (right subplot)
ax2 = plt.subplot(122)
plt.plot(Xo2[:, 0][yo2 == 0], Xo2[:, 1][yo2 == 0], "o")  # Class 0 points
plt.plot(Xo2[:, 0][yo2 == 1], Xo2[:, 1][yo2 == 1], "s")  # Class 1 points
plot_svc_decision_boundary(svm_clf2, 0, 5.5)  # Plot decision boundary and margins
plt.xlabel("petal length (cm)")
plt.axis([0, 5.5, 0, 2])  # Set axis limits

ax2.set_yticklabels([])

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("SVM_hard_margin.pdf")




