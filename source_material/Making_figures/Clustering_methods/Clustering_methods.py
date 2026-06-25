#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP

try:
    IP.get_ipython().run_line_magic('reset', '-sf')
except AttributeError:
    pass

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.

Header file last updated July 19, 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

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

#%% Helper functions

def setup_panel(ax, title):
    """Format one schematic panel."""
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xticks(np.arange(-3, 4, 1))
    ax.set_yticks(np.arange(-2, 3, 1))
    ax.grid(True, linewidth=0.45, alpha=0.35)

    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('0.15')

    ax.text(0.5, -0.18, title,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=10,
            fontweight='bold')


def add_ellipse(ax, center, width, height, angle, color,
                alpha=0.16, lw=1.7, ls='-'):
    """Add a filled covariance/cluster ellipse."""
    patch = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=mpl.colors.to_rgba(color, alpha),
        edgecolor=color,
        linewidth=lw,
        linestyle=ls
    )
    ax.add_patch(patch)
    return patch


def rotated_gaussian(rng, mean, sx, sy, angle_deg, n):
    """Generate a rotated 2D Gaussian cloud."""
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    D = np.diag([sx, sy])
    X = rng.normal(size=(n, 2)) @ D @ R.T
    return X + np.asarray(mean)


#%% Data for schematic panels

rng = np.random.default_rng(11)

# Basic compact clusters used in K-means and hierarchical panels
C1 = rotated_gaussian(rng, [-1.65,  0.70], 0.32, 0.22,  10, 22)
C2 = rotated_gaussian(rng, [ 1.25,  0.70], 0.34, 0.24, -15, 22)
C3 = rotated_gaussian(rng, [-0.35, -1.05], 0.56, 0.28,   5, 32)

# DBSCAN/HDBSCAN-style data: a ring, a dense center, and noise
theta = np.linspace(0, 2*np.pi, 38, endpoint=False)
ring = np.column_stack([
    1.50*np.cos(theta),
    1.00*np.sin(theta)
])
ring += rng.normal(scale=[0.08, 0.08], size=ring.shape)

center = rotated_gaussian(rng, [0.0, 0.0], 0.20, 0.16, 20, 15)

noise = np.array([
    [-2.75,  1.55],
    [-2.55, -1.55],
    [ 2.55,  1.55],
    [ 2.75, -1.45],
    [-2.85,  0.00],
    [ 2.85,  0.15]
])

# Gaussian mixture data
G1 = rotated_gaussian(rng, [-1.15,  0.35], 0.60, 0.22,  30, 34)
G2 = rotated_gaussian(rng, [ 1.10,  0.45], 0.55, 0.24, -25, 34)
G3 = rotated_gaussian(rng, [ 0.10, -1.15], 0.70, 0.20,   0, 32)

# Anomaly detection data
normal = rotated_gaussian(rng, [0.0, 0.0], 0.75, 0.35, -20, 70)
anomalies = np.array([
    [-2.65,  1.45],
    [-2.35, -1.45],
    [ 2.45,  1.35],
    [ 2.65, -1.35],
    [ 0.15,  1.85]
])

#%% Make figure

fig, axs = plt.subplots(1, 5, figsize=(7.4, 2.05), constrained_layout=False)

# -------------------------------------------------------------------------
# Panel 1: K-means
# -------------------------------------------------------------------------
ax = axs[0]
setup_panel(ax, r'K-Means')

ax.scatter(C1[:, 0], C1[:, 1], s=16, color=cc[0], zorder=3)
ax.scatter(C2[:, 0], C2[:, 1], s=16, color=cc[1], zorder=3)
ax.scatter(C3[:, 0], C3[:, 1], s=16, color=cc[2], zorder=3)

for data, color, width, height in [
        (C1, cc[0], 1.25, 0.95),
        (C2, cc[1], 1.25, 0.95),
        (C3, cc[2], 2.25, 1.10)]:
    mu = data.mean(axis=0)
    add_ellipse(ax, mu, width, height, 0, color, alpha=0.12, lw=1.6)
    ax.scatter(mu[0], mu[1],
               marker='+', s=90, linewidths=1.8,
               color=color, zorder=5)

# -------------------------------------------------------------------------
# Panel 2: Hierarchical clustering
# -------------------------------------------------------------------------
ax = axs[1]
setup_panel(ax, r'Hierarchical')

ax.scatter(C1[:, 0], C1[:, 1], s=16, color=cc[0], zorder=3)
ax.scatter(C2[:, 0], C2[:, 1], s=16, color=cc[1], zorder=3)
ax.scatter(C3[:, 0], C3[:, 1], s=16, color=cc[2], zorder=3)

# Small clusters
add_ellipse(ax, C1.mean(axis=0), 1.15, 0.85,  5, cc[0], alpha=0.10, lw=1.5)
add_ellipse(ax, C2.mean(axis=0), 1.15, 0.85, -5, cc[1], alpha=0.10, lw=1.5)
add_ellipse(ax, C3.mean(axis=0), 2.20, 1.00,  0, cc[2], alpha=0.10, lw=1.5)

# Larger nested groups
add_ellipse(ax, [-0.15, 0.55], 4.50, 1.70, 0, cc[4],
            alpha=0.04, lw=1.3, ls='--')
add_ellipse(ax, [-0.20, -0.25], 5.35, 3.30, 0, '0.25',
            alpha=0.02, lw=1.2, ls=':')

# -------------------------------------------------------------------------
# Panel 3: DBSCAN/HDBSCAN
# -------------------------------------------------------------------------
ax = axs[2]
setup_panel(ax, r'DBSCAN/HDBSCAN')

ax.scatter(ring[:, 0], ring[:, 1], s=16, color=cc[2], zorder=3)
ax.scatter(center[:, 0], center[:, 1], s=16, color=cc[1], zorder=3)
ax.scatter(noise[:, 0], noise[:, 1], s=18, color=cc[3], zorder=4)

add_ellipse(ax, [0, 0], 3.35, 2.35, 0, cc[2], alpha=0.06, lw=1.5)
add_ellipse(ax, [0, 0], 0.95, 0.70, 0, cc[1], alpha=0.10, lw=1.5)

# Show a few local neighborhoods
for pt in ring[[2, 12, 24]]:
    add_ellipse(ax, pt, 0.55, 0.55, 0, cc[2], alpha=0.03, lw=0.9, ls='--')

# -------------------------------------------------------------------------
# Panel 4: Gaussian mixture model
# -------------------------------------------------------------------------
ax = axs[3]
setup_panel(ax, r'Gaussian Mixture')

ax.scatter(G1[:, 0], G1[:, 1], s=14, color=cc[0], alpha=0.80, zorder=3)
ax.scatter(G2[:, 0], G2[:, 1], s=14, color=cc[1], alpha=0.80, zorder=3)
ax.scatter(G3[:, 0], G3[:, 1], s=14, color=cc[2], alpha=0.80, zorder=3)

for data, color, angle in [
        (G1, cc[0],  30),
        (G2, cc[1], -25),
        (G3, cc[2],   0)]:
    mu = data.mean(axis=0)
    add_ellipse(ax, mu, 2.30, 0.85, angle, color, alpha=0.08, lw=1.4)
    add_ellipse(ax, mu, 1.35, 0.50, angle, color, alpha=0.10, lw=1.4)
    ax.scatter(mu[0], mu[1],
               marker='+', s=75, linewidths=1.6,
               color=color, zorder=5)

# -------------------------------------------------------------------------
# Panel 5: Anomaly detection
# -------------------------------------------------------------------------
ax = axs[4]
setup_panel(ax, r'Anomaly Detection')

ax.scatter(normal[:, 0], normal[:, 1], s=15, color=cc[0], alpha=0.75, zorder=3)
ax.scatter(anomalies[:, 0], anomalies[:, 1],
           s=24, color=cc[3], marker='x', linewidths=1.8, zorder=5)

add_ellipse(ax, [0, 0], 3.10, 1.40, -20, cc[0], alpha=0.08, lw=1.5)
add_ellipse(ax, [0, 0], 4.30, 1.95, -20, cc[0], alpha=0.04, lw=1.3, ls='--')

#%% Save figure

plt.tight_layout()

plt.savefig('Clustering_methods.jpg',dpi=300)






