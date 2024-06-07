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

m = 200
X = np.random.rand(m,1)-0.5
y = 5 * X**2
y = y + np.random.randn(m,1)/10

tree_reg = sk.tree.DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)

x_model = np.linspace(-0.5,0.5,1000).reshape(-1,1)
y_model = tree_reg.predict(x_model)

plt.figure(figsize=(3,2))
plt.plot(X,y,'.',label='data')
plt.plot(x_model,y_model,'-',label='model')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.tight_layout()
plt.savefig('data and model from python.pdf')

#%% Plot the regression tree

export_graphviz(tree_reg,filled=True,rounded=True,out_file='tree')

s= graphviz.Source.from_file('tree')
s.render('tree from python',format='pdf', view=False)































