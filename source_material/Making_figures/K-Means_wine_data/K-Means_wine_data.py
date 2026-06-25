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

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

#%% Load and prepare the wine data

wine = load_wine()
feature_names = list(wine.feature_names)

feature_pair = ['color_intensity', 'proline']
cols = [feature_names.index(name) for name in feature_pair]

X = wine.data[:, cols]
X = StandardScaler().fit_transform(X)

y = wine.target

x_label = r'$x_1$: color intensity (standardized)'
y_label = r'$x_2$: proline (standardized)'

#%% Plot the data using the known wine classes

fig, ax = plt.subplots(figsize=(4.5, 3.0))

for class_id in np.unique(y):
    mask = y == class_id

    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        s=20,
        color=cc[class_id],
        alpha=0.85,
        linewidths=0,
        label='class ' + str(class_id + 1)
    )

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)


ax.legend(
    framealpha=1,
    loc='best',
    handletextpad=0.4,
    borderpad=0.2,
    labelspacing=0.3
)



#%% Save and show
plt.tight_layout()
plt.savefig('K-means_wine_data.jpg',dpi=300)