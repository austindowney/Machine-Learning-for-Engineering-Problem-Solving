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

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


m = 200
X = np.random.rand(m,1)-0.5
y = 5 * X**2
y = y + np.random.randn(m,1)/10

tree_reg1 = DecisionTreeRegressor(random_state=2, max_depth=20,min_samples_leaf=1)
tree_reg2 = DecisionTreeRegressor(random_state=2, max_depth=20,min_samples_leaf=10)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

plt.figure(figsize=(6.5, 2.5))
ax1 = plt.subplot(121)

x1 = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)
y_pred = tree_reg1.predict(x1)

plt.plot(X, y, ".",label='data')
plt.plot(x1, y_pred, "--", linewidth=2, label=r"$\hat{y}$")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.ylim([-0.21,1.21])

plt.legend(loc="upper center",framealpha=1)
plt.title("min_samples_leaf=1")
plt.grid(True)


ax2 = plt.subplot(122)
y_pred = tree_reg2.predict(x1)

plt.xlabel("$x_1$")
plt.plot(X, y, ".")
plt.plot(x1, y_pred, "--", linewidth=2, label=r"$\hat{y}$")
plt.ylim([-0.21,1.21])

    
plt.title("min_samples_leaf=10")
ax2.set_yticklabels([])
plt.grid(True)

plt.tight_layout()
plt.savefig("decision_tree_regression_regularized",dpi=300)























