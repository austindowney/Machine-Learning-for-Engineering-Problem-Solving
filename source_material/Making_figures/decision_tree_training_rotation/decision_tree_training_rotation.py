import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.

Header file last updated May 16, 2024
"""

import numpy as np
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

np.random.seed(5)

X_square = np.random.rand(100, 2) - 0.5
y_square = (X_square[:, 0] > 0).astype(np.int64)

angle = np.pi / 4  
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)

tree_clf_square = DecisionTreeClassifier(random_state=2)
tree_clf_square.fit(X_square, y_square)

tree_clf_rotated_square = DecisionTreeClassifier(random_state=2)
tree_clf_rotated_square.fit(X_rotated_square, y_square)




#%% Plot the rotation plots


plt.figure(figsize=(6.5, 2.5))
ax1 = plt.subplot(121)
axes=[-0.7, 0.7, -0.7, 0.7]

x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                     np.linspace(axes[2], axes[3], 100))
X_new = np.c_[x1.ravel(), x2.ravel()]
y_pred = tree_clf_square.predict(X_new).reshape(x1.shape)
plt.contourf(x1, x2, y_pred, alpha=0.3, cmap="Pastel2")
plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)
plt.scatter(X_square[y_square == 0, 0],X_square[y_square == 0, 1],marker='s',label="class 0")
plt.scatter(X_square[y_square == 1, 0],X_square[y_square == 1, 1],marker='d',label="class 1")
plt.xlabel("$x_1$")
plt.ylabel(r"$x_2$")
    
    
ax2 = plt.subplot(122)
axes=[-0.7, 0.7, -0.7, 0.7]
x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                     np.linspace(axes[2], axes[3], 100))
X_new = np.c_[x1.ravel(), x2.ravel()]
y_pred = tree_clf_rotated_square.predict(X_new).reshape(x1.shape)
plt.contourf(x1, x2, y_pred, alpha=0.3, cmap="Pastel2")
plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)
plt.scatter(X_rotated_square[y_square == 0, 0],X_square[y_square == 0, 1],marker='s',label="class 0")
plt.scatter(X_rotated_square[y_square == 1, 0],X_square[y_square == 1, 1],marker='d',label="class 1")
plt.xlabel("$x_1$")
plt.tight_layout()
plt.savefig("decision_tree_training_rotation",dpi=300)





