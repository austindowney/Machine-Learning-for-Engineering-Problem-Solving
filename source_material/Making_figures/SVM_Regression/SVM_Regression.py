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

#%% Plot figure

# ----------------------------------------------------------
#   Helpers (no plotting)
# ----------------------------------------------------------
def find_support_vectors(svm_reg, X, y):
    """Indices of samples lying outside the ε-tube."""
    residual = np.abs(y - svm_reg.predict(X))
    return np.where(residual >= svm_reg.epsilon)[0]

def compute_regression_lines(svm_reg, x_grid):
    """Return ŷ, upper, lower margins on a supplied x-grid."""
    y_pred = svm_reg.predict(x_grid)
    return y_pred, y_pred + svm_reg.epsilon, y_pred - svm_reg.epsilon

# ----------------------------------------------------------
#   Data & models
# ----------------------------------------------------------
np.random.seed(2)
m  = 50
X  = 2 * np.random.rand(m, 1)
y  = (4 + 3 * X + np.random.randn(m, 1)).ravel()

svm_reg1 = LinearSVR(epsilon=1.5, random_state=42).fit(X, y)
svm_reg2 = LinearSVR(epsilon=0.5, random_state=42).fit(X, y)

svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

x_grid = np.linspace(0, 2, 100).reshape(-1, 1)

ŷ1, up1, low1 = compute_regression_lines(svm_reg1, x_grid)
ŷ2, up2, low2 = compute_regression_lines(svm_reg2, x_grid)

# Point used to illustrate ε
eps_x1      = 1.0
eps_y_pred1 = svm_reg1.predict([[eps_x1]])[0]

# ----------------------------------------------------------
#   Plotting (subplot style)
# ----------------------------------------------------------
plt.figure(figsize=(6.5, 3))

# ── ε = 1.5 ────────────────────────────────────────────────
plt.subplot(1, 2, 1)
plt.plot(x_grid, ŷ1,  'k-', lw=2, label=r'$\hat{y}$')
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
plt.plot(x_grid, ŷ2,  'k-', lw=2, label=r'$\hat{y}$')
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
plt.savefig("SVM_Regression", dpi=300)
plt.show()
