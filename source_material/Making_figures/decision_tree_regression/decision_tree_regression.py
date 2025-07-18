#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open source notes/
books. This common header is used to set the fonts and format.

Header file last updated March 10, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

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
plt.rcParams.update({'mathtext.fontset': 'custom'}) # I don't think I need this as its set to 'stixsans' above.
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Load the modules needed for this code. 


import matplotlib as mpl
import matplotlib.pyplot as plt


import graphviz as graphviz
from sklearn.tree import DecisionTreeRegressor
import os
from graphviz import Source
import sklearn as sk
from sklearn.tree import export_graphviz






#%% build dataset

np.random.seed(2)
m = 100
X = np.random.rand(m,1)
y = 5 * X
y = y + np.random.randn(m,1)/10

tree_reg = sk.tree.DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)

x_model = np.linspace(0,1,1000).reshape(-1,1)
y_model = tree_reg.predict(x_model)

plt.figure(figsize=(3,2))
plt.plot(X,y,'.',markersize=3,label='data')
plt.plot(x_model,y_model,'--',linewidth=1.5,label='model')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim([-0.4,5.4])
plt.yticks([0,1,2,3,4,5])
plt.legend()
plt.tight_layout()
plt.savefig('data and model from python.pdf')

#%% Plot the regression tree

export_graphviz(tree_reg,filled=True,rounded=True,out_file='tree')

s= graphviz.Source.from_file('tree')

s.render('tree from python',format='svg', view=False) # if set to pdf, it returned an error on office computer - 6/2025
s.render('tree from python',format='jpg', view=False) # if set to pdf, it returned an error on office computer - 6/2025




# --- 1.  Get the DOT source as a *string*  --------------------
dot_data = export_graphviz(
        tree_reg,
        filled=True,
        rounded=True,
        out_file=None)           # ← returns str instead of writing a file

# --- 2.  Build a Graphviz Source object -----------------------
g = graphviz.Source(dot_data, filename="tree")  # "tree" is the base-name

# Optional: tweak global node style so text never overruns the box
g.node_attr.update(
    fontsize='9',               # default is 14; tweak to taste
    margin='0.06,0.04')         # (x-margin, y-margin) in inches

# --- 3.  Render ------------------------------------------------
g.render(format="svg", cleanup=True)   # creates tree.svg (no viewer pop-up)






















