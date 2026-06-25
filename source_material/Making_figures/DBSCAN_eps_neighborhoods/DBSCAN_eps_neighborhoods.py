#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
import sklearn as sk
from sklearn import datasets, cluster, preprocessing

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
plt.rcParams.update({'mathtext.fontset': 'custom'}) 
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Generate make_moons data

X, y = sk.datasets.make_moons(n_samples=500,
                              noise=0.07,
                              random_state=42)

X = sk.preprocessing.StandardScaler().fit_transform(X)

#%% DBSCAN with useful epsilon

eps = 0.20
min_samples = 5

dbscan = sk.cluster.DBSCAN(eps=eps,
                           min_samples=min_samples)

labels = dbscan.fit_predict(X)

core_mask = np.zeros(labels.shape, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

#%% Plot figure

fig, ax = plt.subplots(figsize=(6.5, 3.0))

# Draw epsilon neighborhoods around core points
for ii in dbscan.core_sample_indices_:

    if labels[ii] != -1:

        color = cc[int(labels[ii]) % len(cc)]

        circle = Circle((X[ii, 0], X[ii, 1]),
                        eps,
                        facecolor=mpl.colors.to_rgba(color, 0.08),
                        edgecolor='none',
                        zorder=1)

        ax.add_patch(circle)

# Draw DBSCAN labels
for k in np.unique(labels):

    if k == -1:

        ax.scatter(X[labels == k, 0],
                   X[labels == k, 1],
                   s=24,
                   color=cc[3],
                   marker='x',
                   linewidths=1.2,
                   zorder=4)

    else:

        cluster_core = np.logical_and(labels == k, core_mask)
        cluster_border = np.logical_and(labels == k,
                                        core_mask == False)

        ax.scatter(X[cluster_core, 0],
                   X[cluster_core, 1],
                   s=18,
                   color=cc[int(k) % len(cc)],
                   edgecolors='white',
                   linewidths=0.25,
                   zorder=3)

        ax.scatter(X[cluster_border, 0],
                   X[cluster_border, 1],
                   s=32,
                   color=cc[8],
                   edgecolors=cc[int(k) % len(cc)],
                   linewidths=0.7,
                   zorder=5)

ax.set_title(r'$\epsilon = %.2f$, min samples $= %d$' %
             (eps, min_samples))
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.7, 1.7)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

#%% Save figure
plt.tight_layout()
plt.savefig('DBSCAN_eps_neighborhoods.jpg',dpi=300)