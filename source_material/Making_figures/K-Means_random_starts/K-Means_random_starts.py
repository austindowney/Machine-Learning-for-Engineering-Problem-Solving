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

from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler

#%% Load and prepare the wine data

wine = load_wine()
feature_names = list(wine.feature_names)

feature_pair = ['color_intensity', 'proline']
cols = [feature_names.index(name) for name in feature_pair]

X = wine.data[:, cols]
X = StandardScaler().fit_transform(X)

x_label = r'$x_1$: color intensity (standardized)'
y_label = r'$x_2$: proline (standardized)'

#%% Helper function for plotting decision regions

def plot_kmeans_regions(ax, X, labels, centers, k):
    pad = 0.45

    x_min = X[:, 0].min() - pad
    x_max = X[:, 0].max() + pad
    y_min = X[:, 1].min() - pad
    y_max = X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    region_labels = pairwise_distances_argmin(grid, centers)
    region_labels = region_labels.reshape(xx.shape)

    cmap = ListedColormap(cc[:k])

    ax.contourf(
        xx,
        yy,
        region_labels,
        levels=np.arange(k + 1) - 0.5,
        cmap=cmap,
        alpha=0.22
    )

    ax.contour(
        xx,
        yy,
        region_labels,
        levels=np.arange(1, k) - 0.5,
        colors='black',
        linewidths=0.75,
        alpha=0.55
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap=cmap,
        vmin=-0.5,
        vmax=k - 0.5,
        s=14,
        alpha=0.85,
        linewidths=0
    )

    for j in range(k):
        ax.scatter(
            centers[j, 0],
            centers[j, 1],
            marker='X',
            s=115,
            color=cc[j],
            edgecolor='black',
            linewidth=0.9,
            zorder=5
        )

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


#%% Search for two different random-initialization solutions

k = 3
fits = []

for seed in range(250):
    model = KMeans(
        n_clusters=k,
        init='random',
        n_init=1,
        random_state=seed
    )

    model.fit(X)

    fits.append((
        model.inertia_,
        seed,
        model.labels_,
        model.cluster_centers_
    ))

fits = sorted(fits, key=lambda item: item[0])

best_inertia, best_seed, best_labels, best_centers = fits[0]
worst_inertia, worst_seed, worst_labels, worst_centers = fits[-1]

#%% Plot the two solutions

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.0, 3.0),
    sharex=True,
    sharey=True
)

plot_kmeans_regions(
    axes[0],
    X,
    best_labels,
    best_centers,
    k
)

axes[0].set_title(
    'Lower-inertia start\n'
    + 'seed=' + str(best_seed)
    + ', inertia=' + str(np.round(best_inertia, 1))
)

axes[0].set_xlabel(x_label)
axes[0].set_ylabel(y_label)

plot_kmeans_regions(
    axes[1],
    X,
    worst_labels,
    worst_centers,
    k
)

axes[1].set_title(
    'Higher-inertia start\n'
    + 'seed=' + str(worst_seed)
    + ', inertia=' + str(np.round(worst_inertia, 1))
)

axes[1].set_xlabel(x_label)

#%% Save and show
plt.tight_layout()
plt.savefig('K-Means_random_starts.jpg',dpi=300)