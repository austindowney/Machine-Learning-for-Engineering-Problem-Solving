#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.

Header file last updated July 19, 2025
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

from sklearn.svm import SVC

# === Generate circular dataset ===
np.random.seed(0)
X = np.random.uniform(-1, 1, (200, 2))
y = np.where(X[:, 0]**2 + X[:, 1]**2 > 0.5, 1, -1)

# === Explicit polynomial feature map: (x1^2, sqrt(2)x1x2, x2^2) ===
phi = np.c_[X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2]


# Train a *linear* SVM in feature space to get a true separating plane
clf_linear = SVC(kernel="linear", C=10)
clf_linear.fit(phi, y)
w = clf_linear.coef_[0]
b = clf_linear.intercept_[0]


# === Build figure ===
fig = plt.figure(figsize=(6.5,3.1))

# Left panel: original 2D data
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', s=20)
ax1.scatter(X[y == -1, 0], X[y == -1, 1], facecolors='none', edgecolors=cc[1], s=20)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_xlabel('(a)')

# Right panel: 3D feature space
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

# Data points
ax2.scatter(phi[y == 1, 0], phi[y == 1, 1], phi[y == 1, 2],
            c=cc[0], marker='+', s=20, depthshade=True)


# Separating plane
xx, yy = np.meshgrid(np.linspace(0, 0.6, 20), np.linspace(-1.5, 1.5, 20))
zz = -(w[0]*xx + w[1]*yy + b) / w[2]

ax2.plot_surface(xx, yy, zz,
                  color="gray",
                  alpha=1,
                  linewidth=0.2,
                  rstride=20,
                  edgecolor="k"
                  ,zorder=1)

# ax2.plot_wireframe(xx, yy, zz, 
#                    color="green", 
#                    linewidth=0.5, 
#                    rstride=1, 
#                    cstride=1,
#                    zorder=1)


ax2.scatter(phi[y == -1, 0], phi[y == -1, 1]-0.2, phi[y == -1, 2]+0.0,
            facecolors='none', edgecolors=cc[1], s=20, depthshade=True,
            zorder=10)

# Parabola surface
x1 = np.linspace(-1, 1, 18)
x2 = np.linspace(-1, 1, 18)
X1, X2 = np.meshgrid(x1, x2)

Y1 = X1**2
Y2 = np.sqrt(2) * X1 * X2 + 0.05
Y3 = X2**2

ax2.plot_surface(Y1, Y2, Y3,
                 color="white",
                 edgecolor="k",
                 linewidth=0.8,
                 alpha=0.8,
                 shade=False)



# Fix axis bounds
ax2.set_xlim(0, 1)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(0, 1)


ax2.view_init(elev=5, azim=-133, roll=0)
ax2.view_init(elev=24, azim=-106, roll=0)

fig.text(.78, .01,'(b)')
plt.tight_layout(pad=0)

# Save the figure
#fig.savefig('SVM_nonlinear_illustration', dpi=300, bbox_inches="tight")
fig.savefig('SVM_nonlinear_illustration-just-orange.pdf', dpi=300, bbox_inches="tight")

