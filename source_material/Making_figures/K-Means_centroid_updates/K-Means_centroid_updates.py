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

#%% Helper functions

def make_grid(X, pad=0.45, n_grid=350):
    x_min = X[:, 0].min() - pad
    x_max = X[:, 0].max() + pad
    y_min = X[:, 1].min() - pad
    y_max = X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def update_centers(X, labels, centers):
    new_centers = centers.copy()

    for j in range(centers.shape[0]):
        mask = labels == j

        if np.any(mask):
            new_centers[j] = X[mask].mean(axis=0)

    return new_centers


def lloyd_trace(X, initial_centers, max_iter=5):
    centers = initial_centers.copy()
    centers_trace = [centers.copy()]

    for _ in range(max_iter):
        labels = pairwise_distances_argmin(X, centers)
        new_centers = update_centers(X, labels, centers)

        centers_trace.append(new_centers.copy())

        if np.allclose(new_centers, centers):
            centers = new_centers.copy()

            while len(centers_trace) < max_iter + 1:
                centers_trace.append(centers.copy())

            break

        centers = new_centers.copy()

    return np.array(centers_trace)


def plot_regions(ax, X, centers, k):
    xx, yy, grid = make_grid(X)

    region_labels = pairwise_distances_argmin(grid, centers)
    region_labels = region_labels.reshape(xx.shape)

    point_labels = pairwise_distances_argmin(X, centers)

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
        linewidths=0.65,
        alpha=0.50
    )

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=point_labels,
        cmap=cmap,
        vmin=-0.5,
        vmax=k - 0.5,
        s=8,
        alpha=0.80,
        linewidths=0
    )


#%% Run Lloyd iterations from a random initialization

k = 3
rng = np.random.default_rng(0)

initial_indices = rng.choice(X.shape[0], size=k, replace=False)
initial_centers = X[initial_indices]

centers_trace = lloyd_trace(
    X,
    initial_centers,
    max_iter=5
)

#%% Plot centroid movement across iterations

fig, axes = plt.subplots(
    2,
    3,
    figsize=(7.2, 4.5),
    sharex=True,
    sharey=True
)

axes = axes.ravel()

for t, ax in enumerate(axes):
    centers = centers_trace[t]

    if t == 0:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=8,
            color='black',
            alpha=0.45,
            linewidths=0
        )

        ax.set_title('Initial centroids')
    else:
        plot_regions(ax, X, centers, k)
        ax.set_title('Update ' + str(t))

        for j in range(k):
            path = centers_trace[:t + 1, j, :]

            ax.plot(
                path[:, 0],
                path[:, 1],
                color=cc[j],
                marker='o',
                markersize=2.5,
                linewidth=1.1,
                zorder=6
            )

    for j in range(k):
        ax.scatter(
            centers[j, 0],
            centers[j, 1],
            marker='X',
            s=80,
            color=cc[j],
            edgecolor='black',
            linewidth=0.8,
            zorder=7
        )

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    if t in [3, 4, 5]:
        ax.set_xlabel(x_label)

    if t in [0, 3]:
        ax.set_ylabel(y_label)

#%% Save and show
plt.tight_layout()
plt.savefig('K-means_centroid_updates.jpg',dpi=300)