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
import sklearn as sk
from sklearn import cluster, datasets, utils

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

#%% Load image from scikit-learn

image = sk.datasets.load_sample_image("flower.jpg")
image = np.array(image, dtype=np.float64) / 255.0

height, width, channels = image.shape
X = image.reshape(-1, channels)

# Use a subset of pixels to fit K-means, then predict all pixels.
# This keeps the figure fast to generate while still segmenting the full image.
X_sample = sk.utils.shuffle(X, random_state=42, n_samples=5000)

# Order of panels after the original image
n_colors = [20, 15, 10, 5, 4, 3, 2]

#%% Plot figure

fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.0))
axs = axs.ravel()

# Original image in the top-left panel
axs[0].imshow(image)
axs[0].set_title("Original image", fontsize=9, pad=2)
axs[0].set_xticks([])
axs[0].set_yticks([])

# K-means image segmentation panels
for ii, k in enumerate(n_colors):

    kmeans = sk.cluster.KMeans(n_clusters=k,
                               random_state=42,
                               n_init=1,
                               max_iter=50)

    kmeans.fit(X_sample)

    labels = kmeans.predict(X)
    colors = kmeans.cluster_centers_

    segmented_image = colors[labels]
    segmented_image = segmented_image.reshape(height, width, channels)
    segmented_image = np.clip(segmented_image, 0, 1)

    axs[ii + 1].imshow(segmented_image)
    axs[ii + 1].set_title(str(k) + " colors", fontsize=9, pad=2)
    axs[ii + 1].set_xticks([])
    axs[ii + 1].set_yticks([])

for ax in axs:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

plt.tight_layout(pad=0.35, w_pad=0.4, h_pad=0.6)

#%% Save figure
plt.tight_layout()
plt.savefig('K-Means_Image_segmentation.jpg',dpi=300)

