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


import sklearn as sk
from sklearn.tree import export_graphviz
import graphviz as graphviz


np.random.seed(2)
m = 100
X = np.random.rand(m,1)
y = 5 * X
y = y + np.random.randn(m,1)/10

tree_reg1 = sk.tree.DecisionTreeRegressor(random_state=2, max_depth=1)
tree_reg2 = sk.tree.DecisionTreeRegressor(random_state=2, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

plt.figure(figsize=(6.5, 2.5))
ax1 = plt.subplot(121)

x1 = np.linspace(0, 1, 500).reshape(-1, 1)
y_pred = tree_reg1.predict(x1)

plt.plot(X, y, ".",markersize=3,label='data')
plt.plot(x1, y_pred, "--", linewidth=1.5, label=r"$\hat{y}$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.ylim([-0.4,5.4])
plt.yticks([0,1,2,3,4,5])
#plt.xticks([-0.5,-0.25,0,0.25,0.5])
plt.legend(loc="upper left",framealpha=1)
plt.title("max_depth=1")
plt.grid(True)


ax2 = plt.subplot(122)
y_pred = tree_reg2.predict(x1)

plt.xlabel("$x$")
plt.plot(X, y, ".",markersize=3)
plt.plot(x1, y_pred, "--", linewidth=1.5, label=r"$\hat{y}$")
plt.ylim([-0.4,5.4])
plt.yticks([0,1,2,3,4,5])
#plt.xticks([-0.5,-0.25,0,0.25,0.5])
plt.title("max_depth=3")
ax2.set_yticklabels([])
plt.grid(True)

plt.tight_layout()
plt.savefig("decision_tree_regression_prediction.png",dpi=300)





export_graphviz(tree_reg2,filled=True,rounded=True,out_file='tree')

s= graphviz.Source.from_file('tree')
s.render('tree from python',format='pdf', view=False)



















