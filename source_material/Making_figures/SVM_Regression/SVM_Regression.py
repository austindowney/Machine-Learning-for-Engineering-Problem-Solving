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
import sklearn as sk
from sklearn import svm


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

#%% Plot figure

m  = 50
X  = 2 * np.random.rand(m, 1)
y  = (4 + 3 * X + np.random.randn(m, 1)).ravel()

svm_reg1 = sk.svm.LinearSVR(epsilon=1.5, random_state=42).fit(X, y)
svm_reg2 = sk.svm.LinearSVR(epsilon=0.5, random_state=42).fit(X, y)

svm_reg1.support_ = np.where(np.abs(y - svm_reg1.predict(X)) >= svm_reg1.epsilon)[0]
svm_reg2.support_ = np.where(np.abs(y - svm_reg2.predict(X)) >= svm_reg2.epsilon)[0]

x_grid = np.linspace(0, 2, 100).reshape(-1, 1)

# Model 1: predictions and ±ε margins
y_hat_1 = svm_reg1.predict(x_grid)
up1     = y_hat_1 + svm_reg1.epsilon
low1    = y_hat_1 - svm_reg1.epsilon

# Model 2: predictions and ±ε margins
y_hat_2 = svm_reg2.predict(x_grid)
up2     = y_hat_2 + svm_reg2.epsilon
low2    = y_hat_2 - svm_reg2.epsilon


# Point used to illustrate ε
eps_x1      = 1.0
eps_y_pred1 = svm_reg1.predict([[eps_x1]])[0]

# ----------------------------------------------------------
#   Plotting (subplot style)
# ----------------------------------------------------------
plt.figure(figsize=(6.5, 3))

# ── ε = 1.5 ────────────────────────────────────────────────
plt.subplot(1, 2, 1)
plt.plot(x_grid, y_hat_1,  'k-', lw=2, label=r'$\hat{y}$')
plt.plot(x_grid, up1,  'k--')
plt.plot(x_grid, low1, 'k--')
plt.scatter(X[svm_reg1.support_], y[svm_reg1.support_],
            s=150, facecolors="none", edgecolors=cc[3], linewidths=2,zorder=10)
plt.plot(X, y, 'o', color=cc[0])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$y$")
plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon))
plt.axis([0, 2, 3, 11])
plt.text(0.91, 5.6, r"$\epsilon$")
plt.legend(loc='upper left')

# ── ε = 0.5 ────────────────────────────────────────────────
plt.subplot(1, 2, 2)
plt.plot(x_grid, y_hat_2,  'k-', lw=2, label=r'$\hat{y}$')
plt.plot(x_grid, up2,  'k--')
plt.plot(x_grid, low2, 'k--')
plt.scatter(X[svm_reg2.support_], y[svm_reg2.support_],
            s=150, facecolors="none",
                        edgecolors=cc[3], linewidths=2,zorder=10)
plt.plot(X, y, 'o', color=cc[0])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$y$")
plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon))
plt.axis([0, 2, 3, 11])
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig("SVM_Regression.pdf")
plt.show()




































