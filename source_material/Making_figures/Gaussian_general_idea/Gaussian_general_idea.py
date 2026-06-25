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
from sklearn.mixture import GaussianMixture

#%% Load and prepare the wine data

wine = load_wine()

feature_name = 'color_intensity'
feature_index = wine.feature_names.index(feature_name)

x_raw = wine.data[:, feature_index].reshape(-1, 1)

scaler = StandardScaler()
x = scaler.fit_transform(x_raw)

#%% Fit a one-dimensional Gaussian mixture model

n_components = 3

gmm = GaussianMixture(
    n_components=n_components,
    covariance_type='full',
    random_state=2,
    n_init=20
)

gmm.fit(x)

# The component numbers from a GMM are arbitrary, so sort them left-to-right
# by their fitted means. This makes the plot easier to read.
means = gmm.means_.ravel()
sigmas = np.sqrt(gmm.covariances_.reshape(n_components))
weights = gmm.weights_

sort_index = np.argsort(means)

means = means[sort_index]
sigmas = sigmas[sort_index]
weights = weights[sort_index]

# Responsibilities are the soft membership probabilities.
responsibilities = gmm.predict_proba(x)
responsibilities = responsibilities[:, sort_index]

# Hard labels are used only for coloring the histogram.
hard_labels = np.argmax(responsibilities, axis=1)

#%% Evaluate the fitted Gaussian curves on a grid

x_grid = np.linspace(x.min() - 0.75, x.max() + 0.75, 600)

component_pdfs = np.zeros((n_components, len(x_grid)))

for j in range(n_components):
    component_pdfs[j, :] = sp.stats.norm.pdf(
        x_grid,
        loc=means[j],
        scale=sigmas[j]
    )

weighted_component_pdfs = weights[:, None] * component_pdfs
mixture_pdf = np.sum(weighted_component_pdfs, axis=0)

# Posterior membership probabilities on the grid
grid_responsibilities = weighted_component_pdfs / np.sum(
    weighted_component_pdfs,
    axis=0
)

max_responsibility = np.max(grid_responsibilities, axis=0)

# Regions where no component is overwhelmingly likely
uncertainty_cutoff = 0.75
uncertain_region = max_responsibility < uncertainty_cutoff

# Choose one representative point with high uncertainty.
# Avoid selecting a point in an extreme low-density tail.
valid = mixture_pdf > 0.10 * np.max(mixture_pdf)
entropy = -np.sum(
    grid_responsibilities * np.log(grid_responsibilities + 1e-12),
    axis=0
)
entropy[~valid] = -np.inf

uncertain_index = np.argmax(entropy)
x_uncertain = x_grid[uncertain_index]
r_uncertain = grid_responsibilities[:, uncertain_index]

#%% Plot

fig, ax = plt.subplots(figsize=(6.5, 3.))

bins = np.linspace(x.min() - 0.4, x.max() + 0.4, 22)

for j in range(n_components):
    x_j = x[hard_labels == j, 0]

    ax.hist(
        x_j,
        bins=bins,
        density=True,
        alpha=0.40,
        color=cc[j],
        edgecolor='none',
        label=(
            rf'component {j + 1}: '
            rf'$\mu={means[j]:.2f}$, '
            rf'$\sigma={sigmas[j]:.2f}$'
        )
    )

    ax.plot(
        x_grid,
        component_pdfs[j, :],
        linestyle='--',
        linewidth=1.5,
        color='black'
    )

# Shade regions where the model has high membership uncertainty
start = None
first_uncertain_label = True

for i, is_uncertain in enumerate(uncertain_region):
    if is_uncertain and start is None:
        start = i

    if start is not None and ((not is_uncertain) or i == len(uncertain_region) - 1):
        stop = i

        if x_grid[stop] - x_grid[start] > 0.05:
            ax.axvspan(
                x_grid[start],
                x_grid[stop],
                alpha=0.12,
                color='gray',
                label='uncertain membership region' if first_uncertain_label else None
            )

            first_uncertain_label = False

        start = None

# # Mark a representative uncertain wine
# ax.axvline(
#     x_uncertain,
#     linestyle=':',
#     linewidth=1.4,
#     color='black'
# )

# probability_text = (
#     rf'at $x={x_uncertain:.2f}$' + '\n'
#     rf'$P(C_1 \mid x)={r_uncertain[0]:.2f}$' + '\n'
#     rf'$P(C_2 \mid x)={r_uncertain[1]:.2f}$' + '\n'
#     rf'$P(C_3 \mid x)={r_uncertain[2]:.2f}$'
# )

# ax.text(
#     x_uncertain + 0.12,
#     0.82 * ax.get_ylim()[1],
#     probability_text,
#     fontsize=9,
#     verticalalignment='top',
#     bbox=dict(
#         boxstyle='round',
#         facecolor='white',
#         edgecolor='black',
#         alpha=0.90
#     )
# )

# ax.set_title('Gaussian mixture uncertainty for the wine data')
ax.set_xlabel(r'$x$: color intensity (standardized)')
ax.set_ylabel('density')

ax.legend(
    loc='upper right',
    framealpha=1
)

fig.tight_layout()


fig.savefig('Gaussian_general_idea.jpg', dpi=300)
