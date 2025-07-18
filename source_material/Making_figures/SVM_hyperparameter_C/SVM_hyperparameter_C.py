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


#%% Load the modules needed for this code. 


from sklearn import datasets, preprocessing, svm


# %% Load data (virginica = 1, versicolor = 0) ---------------------------------
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]               # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)

# %% Two LinearSVC models with different C ------------------------------------
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

clf1 = svm.LinearSVC(C=1,   loss="hinge", random_state=42).fit(X_scaled, y)
clf2 = svm.LinearSVC(C=200, loss="hinge", random_state=42).fit(X_scaled, y)

# %% Convert scaled parameters back to original space -------------------------
def unscale(clf):
    """Return (w, b) in original feature space."""
    w_scaled = clf.coef_[0]
    b_scaled = clf.intercept_[0]
    w_orig = w_scaled / scaler.scale_
    b_orig = b_scaled - np.dot(w_scaled, scaler.mean_ / scaler.scale_)
    return w_orig, b_orig

w1, b1 = unscale(clf1)
w2, b2 = unscale(clf2)

# Identify support vectors manually (LinearSVC does not store them)
t = y * 2 - 1                                # {-1, +1} labels
sv_idx1 = (t * (X @ w1 + b1) < 1).ravel()
sv_idx2 = (t * (X @ w2 + b2) < 1).ravel()
SV1 = X[sv_idx1]
SV2 = X[sv_idx2]

# %% Pre-compute decision boundaries & margins --------------------------------
def decision_line(w, b, xmin, xmax):
    x0 = np.linspace(xmin, xmax, 200)
    x1 = -w[0] / w[1] * x0 - b / w[1]
    margin = 1 / w[1]
    return x0, x1, x1 + margin, x1 - margin

xmin, xmax = 2.8, 6.7
x0, y_dec1, y_up1, y_dn1 = decision_line(w1, b1, xmin, xmax)
_,  y_dec2, y_up2, y_dn2 = decision_line(w2, b2, xmin, xmax)

# %% Plot ---------------------------------------------------------------------
plt.figure(figsize=(6.5, 3.0))

# -- Subplot (a): C = 1 --------------------------------------------------------
plt.subplot(1, 2, 1)
plt.plot(X[y == 0, 0], X[y == 0, 1], "s", color=cc[1],
         label="versicolor",zorder=10)
plt.plot(X[y == 1, 0], X[y == 1, 1], "d", color=cc[2],
         label="virginica",zorder=10)
plt.scatter(SV1[:, 0], SV1[:, 1], s=150, facecolors="none",
            edgecolors=cc[3], linewidths=2,zorder=10)

plt.plot(x0, y_dec1, "k-",  linewidth=2, label="decision boundary")
plt.plot(x0, y_up1,  "k--", linewidth=1)
plt.plot(x0, y_dn1,  "k--", linewidth=1)

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("$C = 1$")
plt.legend(framealpha=1,loc='upper left',fontsize=9)
plt.axis([xmin, xmax, 0.8, 2.8])

# -- Subplot (b): C = 200 ------------------------------------------------------
plt.subplot(1, 2, 2)
plt.plot(X[y == 0, 0], X[y == 0, 1], "s", color=cc[1],zorder=10)
plt.plot(X[y == 1, 0], X[y == 1, 1], "d", color=cc[2],zorder=10)
plt.scatter(SV2[:, 0], SV2[:, 1], s=150, facecolors="none",
            edgecolors=cc[3], linewidths=2,zorder=10)

plt.plot(x0, y_dec2, "k-",  linewidth=2)
plt.plot(x0, y_up2,  "k--", linewidth=1)
plt.plot(x0, y_dn2,  "k--", linewidth=1)

plt.xlabel("petal length (cm)")
plt.title("$C = 200$")
plt.axis([xmin, xmax, 0.8, 2.8])
plt.gca().set_yticklabels([])  # hide y-tick labels on second subplot

plt.tight_layout()
plt.savefig("SVM_hyperparameter_C.png", dpi=300)
plt.show()













