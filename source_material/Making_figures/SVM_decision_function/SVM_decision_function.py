#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.

Header file last updated May 16, 2024
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
# I don't think I need this next line as its set to 'stixsans' above. 
plt.rcParams.update({'mathtext.fontset': 'custom'}) 
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Load special modules as needed

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC



#%% SVM polynominal features


# Load iris dataset
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # Use petal length and petal width
y = (iris["target"] == 2).astype(np.float64)  # Binary classification for Iris virginica

# Define and train SVM classifier with scaling
svm_clf = Pipeline([
    ("scaler", StandardScaler()),  # Feature scaling
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),  # SVM classifier
])
svm_clf.fit(X, y)  # Fit the pipeline on the data

# Define separate SVM classifiers for comparison with different regularization parameters
scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

# Create pipelines for the classifiers
scaled_svm_clf1 = Pipeline([
    ("scaler", scaler),
    ("linear_svc", svm_clf1),
])
scaled_svm_clf2 = Pipeline([
    ("scaler", scaler),
    ("linear_svc", svm_clf2),
])

# Fit the classifiers on the data
scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)

# Convert parameters to unscaled versions
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

# Find support vectors (LinearSVC does not do this automatically)
t = y * 2 - 1  # Convert labels to +1 and -1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()  # Support vectors for svm_clf1
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()  # Support vectors for svm_clf2
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]

# Plotting the decision function in 3D without a function
fig = plt.figure(figsize=(6.5, 4))
ax1 = fig.add_subplot(111, projection='3d')

# Crop the dataset to the x1 limits
x1_lim = [4, 6]
x2_lim = [0.8, 2.8]
x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])
X_crop = X[x1_in_bounds]
y_crop = y[x1_in_bounds]

# Generate a mesh grid for plotting
x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
x1, x2 = np.meshgrid(x1s, x2s)
xs = np.c_[x1.ravel(), x2.ravel()]
df = (xs.dot(w2) + b2).reshape(x1.shape)

# Calculate decision boundary and margins
m = 1 / np.linalg.norm(w2)
boundary_x2s = -x1s * (w2[0] / w2[1]) - b2 / w2[1]
margin_x2s_1 = -x1s * (w2[0] / w2[1]) - (b2 - 1) / w2[1]
margin_x2s_2 = -x1s * (w2[0] / w2[1]) - (b2 + 1) / w2[1]

# Plot the decision function and margins in 3D
ax1.plot_surface(x1s, x2, np.zeros_like(x1), color="b", alpha=0.2, cstride=100, rstride=100)
ax1.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
ax1.plot(x1s, margin_x2s_1, 0, "k:", linewidth=2, label=r"$h=\pm 1$")
ax1.plot(x1s, margin_x2s_2, 0, "k:", linewidth=2)
ax1.plot(X_crop[:, 0][y_crop == 0], X_crop[:, 1][y_crop == 0], 0, "s",color=cc[1], markersize=5)
ax1.plot(X_crop[:, 0][y_crop == 1], X_crop[:, 1][y_crop == 1], 0, "d",color=cc[2], markersize=5)
ax1.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
ax1.axis(x1_lim + x2_lim)
#ax1.text(4.5, 2.6, 3.8, "decision function $h$")
ax1.set_xlabel(r"petal length (cm)", labelpad=5)
ax1.set_ylabel(r"petal width (cm)", labelpad=5)
ax1.set_zlabel(r"$h = \mathbf{w}^T \mathbf{x} + b$", labelpad=0)
ax1.legend(loc="upper left")

plt.tight_layout()
plt.savefig("SVM_decision_function.pdf")















