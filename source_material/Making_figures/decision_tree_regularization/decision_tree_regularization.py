
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

np.random.seed(2)


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=60, noise=0.25)

deep_tree_clf1 = DecisionTreeClassifier()
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5)
deep_tree_clf1.fit(X, y)
deep_tree_clf2.fit(X, y)


#%% plot

plt.figure(figsize=(6.5, 2.5))
ax1 = plt.subplot(121)


axes=[-1.5, 2.4, -1, 1.5]
x1s = np.linspace(axes[0], axes[1], 100)
x2s = np.linspace(axes[2], axes[3], 100)
x1, x2 = np.meshgrid(x1s, x2s)
X_new = np.c_[x1.ravel(), x2.ravel()]
y_pred = deep_tree_clf1.predict(X_new).reshape(x1.shape)
plt.contourf(x1, x2, y_pred, alpha=0.3, cmap="Pastel2")
plt.contour(x1, x2, y_pred, cmap="gray", alpha=0.8)
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o",markersize=4)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "s",markersize=4)
plt.axis(axes)
plt.title("no restrictions")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


ax1 = plt.subplot(122)
axes=[-1.5, 2.4, -1, 1.5]
x1s = np.linspace(axes[0], axes[1], 100)
x2s = np.linspace(axes[2], axes[3], 100)
x1, x2 = np.meshgrid(x1s, x2s)
X_new = np.c_[x1.ravel(), x2.ravel()]
y_pred = deep_tree_clf2.predict(X_new).reshape(x1.shape)
plt.contourf(x1, x2, y_pred, alpha=0.3, cmap="Pastel2")
plt.contour(x1, x2, y_pred, cmap="gray", alpha=0.8)
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o",markersize=4)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "s",markersize=4)
plt.axis(axes)
plt.title("min_samples_leaf = 5")
plt.xlabel("$x_1$")

plt.tight_layout()
plt.savefig("decision_tree_regularization.jpg",dpi=500)












