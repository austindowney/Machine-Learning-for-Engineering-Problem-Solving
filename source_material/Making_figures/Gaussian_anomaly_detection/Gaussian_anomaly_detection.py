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
from matplotlib.patches import Ellipse
import sklearn as sk
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

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

#%% Load and prepare the wine data

wine = load_wine()
feature_names = list(wine.feature_names)

feature_pair = ['color_intensity', 'proline']
cols = [feature_names.index(name) for name in feature_pair]

X = wine.data[:, cols]
X = StandardScaler().fit_transform(X)

x_label = r'$x_1$: color intensity (standardized)'
y_label = r'$x_2$: proline (standardized)'

x_min = X[:, 0].min() - 0.75
x_max = X[:, 0].max() + 0.75
y_min = X[:, 1].min() - 0.75
y_max = X[:, 1].max() + 0.75

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                     np.linspace(y_min, y_max, 250))

X_grid = np.c_[xx.ravel(), yy.ravel()]

#%% Fit Gaussian mixture model

gmm = GaussianMixture(n_components=3,
                      covariance_type='full',
                      random_state=42,
                      n_init=10)

labels = gmm.fit_predict(X)

log_density_grid = gmm.score_samples(X_grid).reshape(xx.shape)
density_grid = -log_density_grid

labels_grid = gmm.predict(X_grid).reshape(xx.shape)

levels = np.linspace(np.nanpercentile(density_grid, 5),
                     np.nanpercentile(density_grid, 95),
                     22)

#%% Identify low-likelihood observations

log_density = gmm.score_samples(X)

threshold = np.percentile(log_density, 5)

outlier_mask = log_density < threshold
normal_mask = ~outlier_mask

threshold_density = -threshold

#%% Plot Gaussian anomaly detection

fig, ax = plt.subplots(figsize=(4.5, 3.0))

ax.contourf(xx, yy, density_grid,
            levels=levels,
            cmap='viridis',
            alpha=0.75)

ax.contour(xx, yy, density_grid,
           levels=levels[::2],
           colors='k',
           linewidths=0.55,
           alpha=0.75)

ax.contour(xx, yy, labels_grid,
           levels=[0.5, 1.5],
           colors=cc[3],
           linewidths=1.4,
           linestyles='--')

ax.scatter(X[normal_mask, 0],
           X[normal_mask, 1],
           s=18,
           color='k',
           alpha=0.55,
           linewidths=0,
           label='normal')

ax.scatter(X[outlier_mask, 0],
           X[outlier_mask, 1],
           s=34,
           color=cc[3],
           marker='x',
           linewidths=1.5,
           label='low likelihood')

ax.contour(xx, yy, density_grid,
           levels=[threshold_density],
           colors=cc[3],
           linewidths=1.7)

for j in range(3):

    mean = gmm.means_[j]
    cov = gmm.covariances_[j]

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0],
                                  eigvecs[0, 0]))

    width = 2 * 2.0 * np.sqrt(eigvals[0])
    height = 2 * 2.0 * np.sqrt(eigvals[1])

    ellipse = Ellipse(mean,
                      width,
                      height,
                      angle=angle,
                      facecolor='none',
                      edgecolor=cc[j],
                      linewidth=1.5)

    ax.add_patch(ellipse)

ax.set_title('Gaussian mixture anomaly detection')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True)

ax.legend(framealpha=1,
          loc='best',
          handletextpad=0.4,
          borderpad=0.2,
          labelspacing=0.3,
          fontsize=8)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

#%% Save figure
plt.tight_layout()
plt.savefig('Gaussian_anomaly_detection.jpg',dpi=300)