
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


#%% Common imports

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier




#%% build model


np.random.seed(2)

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)






#%% plot

plt.figure(figsize=(6.5, 3.5))

axes=[0, 7.5, 0, 3]
x1s = np.linspace(axes[0], axes[1], 100)
x2s = np.linspace(axes[2], axes[3], 100)
x1, x2 = np.meshgrid(x1s, x2s)
X_new = np.c_[x1.ravel(), x2.ravel()]
y_pred = tree_clf.predict(X_new).reshape(x1.shape)
plt.contourf(x1, x2, y_pred, alpha=0.3, cmap='Pastel2')
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", label="setosa")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "s", label="versicolor")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "d", label="virginica")
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(2.2, 1.0, "Depth=0", rotation=90)
plt.text(3.0, 1.85, "Depth=1")
plt.text(4.7, 0.25, "Depth=2", rotation=90)
plt.text(4.6, 2.15, "Depth=2", rotation=90)

plt.text(0.75, 1.4, "predicted\n   setosa")
plt.text(3.3, 0.3, " predicted\nversicolor")
plt.text(5.8, 0.3, "predicted\n virginica")
plt.text(3.3, 2.65, "predicted\n virginica")
plt.text(5.8, 2.65, "predicted\n virginica")

plt.legend(loc="upper left", framealpha=1)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.tight_layout()
plt.savefig("decision_tree_boundaries",dpi=300)



