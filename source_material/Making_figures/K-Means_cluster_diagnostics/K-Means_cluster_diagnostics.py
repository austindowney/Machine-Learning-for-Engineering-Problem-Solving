#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import IPython as IP
    ip = IP.get_ipython()
    if ip is not None:
        ip.run_line_magic('reset', '-sf')
except Exception:
    pass

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
plt.rcParams.update({'mathtext.fontset': 'custom'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Load special modules as needed

from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#%% Load and prepare the wine data

wine = load_wine()
feature_names = list(wine.feature_names)

feature_pair = ['color_intensity', 'proline']
cols = [feature_names.index(name) for name in feature_pair]

X = wine.data[:, cols]
X = StandardScaler().fit_transform(X)

#%% Compute inertia and silhouette scores

k_values = np.arange(1, 9)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=50,
        random_state=8
    )

    model.fit(X)
    inertias.append(model.inertia_)

silhouette_k_values = np.arange(2, 9)
silhouettes = []

for k in silhouette_k_values:
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=50,
        random_state=8
    )

    model.fit(X)
    score = silhouette_score(X, model.labels_)
    silhouettes.append(score)

#%% Plot the elbow rule and silhouette diagnostic

fig, ax1 = plt.subplots(figsize=(6.5, 3.0))

plt.grid('true')
line1, = ax1.plot(
    k_values,
    inertias,
    marker='o',
    color=cc[0],
    label='inertia'
)

ax1.set_xlabel(r'number of clusters, $k$')
ax1.set_ylabel('inertia')



ax2 = ax1.twinx()

line2, = ax2.plot(
    silhouette_k_values,
    silhouettes,
    marker='s',
    color=cc[1],
    label='silhouette'
)

ax2.set_ylabel('mean silhouette score')

for spine in ['top']:
    ax2.spines[spine].set_visible(False)

ax1.legend(
    handles=[line1, line2],
    loc='upper right',
    framealpha=1
)

#%% Save and show
plt.tight_layout()
plt.savefig('K-Means_cluster_diagnostics.jpg',dpi=300)