#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 5.2 Decision Tree Regression

@author: Austin R.J. Downey
"""

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import graphviz as graphviz
from sklearn.tree import export_graphviz

plt.close('all')


#%% Train and plot a decision tree regression model

# build the data
m = 200
X = np.random.rand(m, 1)
y = 5 * X
y = y + np.random.randn(m, 1) / 10

# train the model
tree_reg = sk.tree.DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X, y)

x_model = np.linspace(0, 1, 100).reshape(-1, 1)
y_model = tree_reg.predict(x_model)

plt.figure()
plt.plot(X, y, ".",label='data')
plt.plot(x_model, y_model, "-", label="model")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()

#%% Plot the regression tree

# create the export file for graphviz and export it. The file is exported as a 
# .DOT file and can be viewed in an online viewer https://dreampuf.github.io/GraphvizOnline/
export_graphviz(
        tree_reg,
        out_file="tree_reg",
        rounded=True,
        filled=True
    )

# We can load the file back in
s = graphviz.Source.from_file('tree_reg')
s.render('tree_reg', format='jpg',view=True)




