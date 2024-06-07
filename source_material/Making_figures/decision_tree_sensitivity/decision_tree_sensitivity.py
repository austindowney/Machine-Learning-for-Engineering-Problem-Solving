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
from sklearn.tree import DecisionTreeClassifier


#%% build the models


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target


lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))
X_iris_all = np.c_[lengths.ravel(), widths.ravel()]


#%% Train the models with different seeds

tree_clf_1 = DecisionTreeClassifier(max_depth=2, random_state=1)
tree_clf_1.fit(X, y)
y_pred_1 = tree_clf_1.predict(X_iris_all).reshape(lengths.shape)

tree_clf_2 = DecisionTreeClassifier(max_depth=2, random_state=2)
tree_clf_2.fit(X, y)
y_pred_2 = tree_clf_2.predict(X_iris_all).reshape(lengths.shape)

#%% plot the models

plt.figure(figsize=(6.5, 4.5))


plt.subplot(211)
plt.contourf(lengths, widths, y_pred_1, alpha=0.3, cmap='Pastel2')
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", label="setosa")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "s", label="versicolor")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "d", label="virginica")
th0, th1 = tree_clf_1.tree_.threshold[[0, 2]]
plt.plot([0, 7.2], [th0, th0], "k-", linewidth=2)
plt.plot([0, 7.2], [th1, th1], "k--", linewidth=2)
plt.text(1.8, th0 + 0.05, "depth=0", verticalalignment="bottom")
plt.text(2.3, th1 + 0.05, "depth=1", verticalalignment="bottom")
plt.xlabel("petal length (cm) \n (b)")
plt.ylabel("petal width (cm)")
plt.axis([0, 7.2, 0, 3])


plt.subplot(212)
plt.contourf(lengths, widths, y_pred_2, alpha=0.3, cmap='Pastel2')
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", label="setosa")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "s", label="versicolor")
plt.plot(X[:, 0][y==2], X[:, 1][y==2], "d", label="virginica")
th0, th1 = tree_clf_2.tree_.threshold[[0, 2]]
plt.plot([th0, th0], [0, 3], "k-", linewidth=2)
plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)
plt.text(2.25, 1.8, "depth=0", verticalalignment="bottom",rotation=90)
plt.text(2.8, th1 + 0.05, "depth=1", verticalalignment="bottom")
plt.xlabel("petal length (cm) \n (a)")
plt.ylabel("petal width (cm)")
plt.axis([0, 7.2, 0, 3])
plt.legend()   


plt.tight_layout()

plt.savefig("decision_tree_sensitivity",dpi=300)
















