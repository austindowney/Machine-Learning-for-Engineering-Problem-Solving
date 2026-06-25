#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.
"""

import os
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

os.makedirs("../figures", exist_ok=True)

#%% Figure 1: K-means versus DBSCAN on non-convex data

X_moons, y_moons = sk.datasets.make_moons(n_samples=500,
                                          noise=0.07,
                                          random_state=42)

X_circles, y_circles = sk.datasets.make_circles(n_samples=500,
                                                noise=0.04,
                                                factor=0.45,
                                                random_state=42)

X_moons = sk.preprocessing.StandardScaler().fit_transform(X_moons)
X_circles = sk.preprocessing.StandardScaler().fit_transform(X_circles)

kmeans_moons = sk.cluster.KMeans(n_clusters=2,
                                 random_state=42,
                                 n_init=10)

kmeans_circles = sk.cluster.KMeans(n_clusters=2,
                                   random_state=42,
                                   n_init=10)

dbscan_moons = sk.cluster.DBSCAN(eps=0.25,
                                 min_samples=5)

dbscan_circles = sk.cluster.DBSCAN(eps=0.25,
                                   min_samples=5)

labels_kmeans_moons = kmeans_moons.fit_predict(X_moons)
labels_dbscan_moons = dbscan_moons.fit_predict(X_moons)

labels_kmeans_circles = kmeans_circles.fit_predict(X_circles)
labels_dbscan_circles = dbscan_circles.fit_predict(X_circles)

fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.1))

# K-means on moons
ax = axs[0, 0]
for k in np.unique(labels_kmeans_moons):
    ax.scatter(X_moons[labels_kmeans_moons == k, 0],
               X_moons[labels_kmeans_moons == k, 1],
               s=14, color=cc[int(k) % len(cc)],
               edgecolors='white', linewidths=0.25)
ax.set_title(r'\textbf{K-Means}')
ax.set_ylabel(r'\texttt{make\_moons}')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.7, 1.7)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linewidth=0.4, alpha=0.45)
ax.set_xticks([])
ax.set_yticks([])

# DBSCAN on moons
ax = axs[0, 1]
for k in np.unique(labels_dbscan_moons):
    if k == -1:
        ax.scatter(X_moons[labels_dbscan_moons == k, 0],
                   X_moons[labels_dbscan_moons == k, 1],
                   s=18, color=cc[3], marker='x',
                   linewidths=1.1)
    else:
        ax.scatter(X_moons[labels_dbscan_moons == k, 0],
                   X_moons[labels_dbscan_moons == k, 1],
                   s=14, color=cc[int(k) % len(cc)],
                   edgecolors='white', linewidths=0.25)
ax.set_title(r'\textbf{DBSCAN}')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.7, 1.7)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linewidth=0.4, alpha=0.45)
ax.set_xticks([])
ax.set_yticks([])

# K-means on circles
ax = axs[1, 0]
for k in np.unique(labels_kmeans_circles):
    ax.scatter(X_circles[labels_kmeans_circles == k, 0],
               X_circles[labels_kmeans_circles == k, 1],
               s=14, color=cc[int(k) % len(cc)],
               edgecolors='white', linewidths=0.25)
ax.set_ylabel(r'\texttt{make\_circles}')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linewidth=0.4, alpha=0.45)
ax.set_xticks([])
ax.set_yticks([])

# DBSCAN on circles
ax = axs[1, 1]
for k in np.unique(labels_dbscan_circles):
    if k == -1:
        ax.scatter(X_circles[labels_dbscan_circles == k, 0],
                   X_circles[labels_dbscan_circles == k, 1],
                   s=18, color=cc[3], marker='x',
                   linewidths=1.1)
    else:
        ax.scatter(X_circles[labels_dbscan_circles == k, 0],
                   X_circles[labels_dbscan_circles == k, 1],
                   s=14, color=cc[int(k) % len(cc)],
                   edgecolors='white', linewidths=0.25)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linewidth=0.4, alpha=0.45)
ax.set_xticks([])
ax.set_yticks([])

for ax in axs.ravel():
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

#%% Save and show

plt.tight_layout()

plt.savefig('DBSCAN_vs_k-means.jpg',dpi=300)

