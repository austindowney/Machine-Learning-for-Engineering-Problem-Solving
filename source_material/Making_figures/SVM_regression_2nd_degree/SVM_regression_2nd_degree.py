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

from sklearn.svm import SVR

np.random.seed(2)
m = 100
X = 2 * np.random.rand(m, 1) - 1
y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1) / 10).ravel()

svr1 = SVR(kernel="poly", degree=2, C=100000,  epsilon=0.1, gamma="scale").fit(X, y)
svr2 = SVR(kernel="poly", degree=2, C=0.002, epsilon=0.1, gamma="scale").fit(X, y)

x_grid = np.linspace(-1, 1, 100).reshape(-1, 1)

# Predict and calculate margins for svr1
y_hat_1 = svr1.predict(x_grid)
up1 = y_hat_1 + svr1.epsilon
low1 = y_hat_1 - svr1.epsilon

# Predict and calculate margins for svr2
y_hat_2 = svr2.predict(x_grid)
up2 = y_hat_2 + svr2.epsilon
low2 = y_hat_2 - svr2.epsilon


sup1 = svr1.support_
sup2 = svr2.support_


#%%


plt.figure(figsize=(6.5, 3))


plt.subplot(1, 2, 1)
plt.plot(x_grid, y_hat_1,   'k-', lw=2, label=r'$\hat{y}$')
plt.plot(x_grid, up1,  'k--')
plt.plot(x_grid, low1, 'k--')
plt.scatter(X[sup1], y[sup1], s=80, facecolors="none",
            edgecolors=cc[3], linewidths=1.5,zorder=10)
plt.plot(X, y, 'o', ms=3,zorder=10)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$y$")
plt.title('$C=100$,000')
plt.axis([-1, 1, 0, 1])
plt.legend(loc='upper left')


plt.subplot(1, 2, 2)
plt.plot(x_grid, y_hat_2,   'k-', lw=2, label=r'$\hat{y}$')
plt.plot(x_grid, up2,  'k--')
plt.plot(x_grid, low2, 'k--')
plt.scatter(X[sup2], y[sup2], s=80, facecolors="none",
            edgecolors=cc[3], linewidths=1.5,zorder=10)
plt.plot(X, y, 'o', ms=3,zorder=10)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$y$")
plt.title(r"$C={}$".format(svr2.C))
plt.axis([-1, 1, 0, 1])
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig("SVM_regression_2nd_degree", dpi=300)
plt.show()
