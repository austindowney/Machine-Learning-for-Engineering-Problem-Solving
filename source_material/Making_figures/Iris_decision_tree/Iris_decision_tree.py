
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
from sklearn.datasets import load_iris
import graphviz as graphviz

from sklearn.tree import export_graphviz

plt.close('all')




#%% Train and plot a decision tree classifier

# set a seed
np.random.seed(2)

# loat the data
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

# train the decision tree
tree_clf = sk.tree.DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)

# create the export file for graphviz and export it. The file is exported as a 
# .DOT file and can be viewed in an online viewer https://dreampuf.github.io/GraphvizOnline/
export_graphviz(
        tree_clf,
        out_file="iris_tree.dot",
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )


# We can load the file back in
s = graphviz.Source.from_file('iris_tree.dot')



# export the image to a jpg
s.render('Iris_decision_tree', format='jpg',view=True)





