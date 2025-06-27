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

from sklearn import datasets
from sklearn.svm import SVC

# Use Times New Roman in all plots
mpl.rcParams["font.family"] = "Times New Roman"

# %% Load data (only Setosa and Versicolor) ------------------------------------
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]          # petal length (x1), petal width (x2)
y = iris["target"]

mask = (y == 0) | (y == 1)           # keep only classes 0 and 1
X, y = X[mask], y[mask]

# %% Train linear SVM ----------------------------------------------------------
svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X, y)

# %% “Bad” decision lines for comparison --------------------------------------
x0 = np.linspace(0, 5.5, 200)
bad_line_1 = x0 - 1.8
bad_line_2 = 0.1 * x0 + 0.5

# Retrieve model parameters
w = svm_clf.coef_[0]
b = svm_clf.intercept_[0]

# Decision boundary: w0*x0 + w1*x1 + b = 0  →  x1 = -w0/w1 * x0 - b/w1
decision_boundary = -w[0] / w[1] * x0 - b / w[1]

# Margin lines (distance = 1/‖w‖ in feature space)
margin = 1 / w[1]
gutter_up   = decision_boundary + margin
gutter_down = decision_boundary - margin


# %% Prepare figure ------------------------------------------------------------
plt.figure(figsize=(6.5, 2.5))

# -- Subplot (a): bad models ---------------------------------------------------
plt.subplot(1, 2, 1)
plt.plot(X[y == 0, 0], X[y == 0, 1], "o", label="setosa")
plt.plot(X[y == 1, 0], X[y == 1, 1], "s", label="versicolor")

plt.plot(x0, bad_line_1, "--")
plt.plot(x0, bad_line_2, "--")

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend(loc="upper left", framealpha=1)
plt.axis([0, 5.5, 0, 2])

# -- Subplot (b): SVM decision boundary ---------------------------------------
plt.subplot(1, 2, 2)


# Plot data points
plt.plot(X[y == 0, 0], X[y == 0, 1], "o")
plt.plot(X[y == 1, 0], X[y == 1, 1], "s")

# Plot decision boundary and margins
plt.plot(x0, decision_boundary, "k-",  linewidth=2, label="decision boundary")
plt.plot(x0, gutter_up,        "k--", linewidth=1)
plt.plot(x0, gutter_down,      "k--", linewidth=1)

# Highlight support vectors
sv = svm_clf.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="k", linewidths=1.2, label="support vectors")

plt.xlabel("petal length (cm)")
plt.axis([0, 5.5, 0, 2])
plt.gca().set_yticklabels([])  # hide y-tick labels

plt.tight_layout()
plt.savefig("SVM_large_margin_classification.png", dpi=300)




